//! Credit-risk EAD / CCF / STD RW (L1) — pure kernel.
//!
//! Config-driven formulas aligned with RiskAuthority 7.4.1 semantics
//! (`OUTSTANDING`, `OUT_PLUS_CCF_UNDRAWN`, `PRINCIPAL_ONLY`, `IMPORT_EPE`).
//! No CSR write-back; callers materialize via SQL / CSV.

/// Closed set of EAD expression kinds (no arbitrary SQL).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprKind {
    Outstanding,
    OutPlusCcfUndrawn,
    PrincipalOnly,
    ImportEpe,
}

impl ExprKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_uppercase().as_str() {
            "OUTSTANDING" => Some(Self::Outstanding),
            "OUT_PLUS_CCF_UNDRAWN" | "OUT+CCF*UNDRAWN" => Some(Self::OutPlusCcfUndrawn),
            "PRINCIPAL_ONLY" => Some(Self::PrincipalOnly),
            "IMPORT_EPE" | "IMPORT" => Some(Self::ImportEpe),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EadFormulaRow {
    pub formula_id: String,
    pub supervisor: String,
    pub product_kind: String,
    pub bs_type: String,
    pub book_type: String,
    pub expr_kind: ExprKind,
    pub effective_from: i64,
    pub effective_to: i64,
}

#[derive(Debug, Clone)]
pub struct CcfMapRow {
    pub map_id: String,
    pub supervisor: String,
    pub ccf_rating: String,
    pub ccf_value: f64,
    pub product_kind: String,
    pub bs_type: String,
    pub effective_from: i64,
    pub effective_to: i64,
}

#[derive(Debug, Clone)]
pub struct RwMapRow {
    pub map_id: String,
    pub supervisor: String,
    pub approach: String,
    pub claim_type: String,
    pub rating: String,
    pub product_kind: String,
    pub rw: f64,
    pub effective_from: i64,
    pub effective_to: i64,
}

#[derive(Debug, Clone)]
pub struct SupervisorOptions {
    pub supervisor: String,
    pub crm_pre_ccf_std: bool,
    pub crm_pre_ccf_irb: bool,
    pub default_ead_formula_id: String,
    pub ccf_explicit_wins: bool,
    pub approach: String,
    pub profile: String,
    pub effective_from: i64,
    pub effective_to: i64,
}

impl Default for SupervisorOptions {
    fn default() -> Self {
        Self {
            supervisor: "HKMA".into(),
            crm_pre_ccf_std: true,
            crm_pre_ccf_irb: false,
            default_ead_formula_id: String::new(),
            ccf_explicit_wins: true,
            approach: "STD".into(),
            profile: "ray8_compat".into(),
            effective_from: 0,
            effective_to: i64::MAX,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CrEadConfig {
    pub formulas: Vec<EadFormulaRow>,
    pub ccf_maps: Vec<CcfMapRow>,
    pub rw_maps: Vec<RwMapRow>,
    pub options: Vec<SupervisorOptions>,
}

/// One exposure input row (L0 / loan_exposure shaped).
#[derive(Debug, Clone, Default)]
pub struct ExposureInput {
    pub loan_id: u64,
    pub outstanding: f64,
    pub undrawn: f64,
    pub accrued_interest: f64,
    pub fees: f64,
    pub ead: f64,
    pub epe: f64,
    pub ccf: Option<f64>,
    pub ccf_rating: String,
    pub rating: String,
    pub product_kind: String,
    pub bs_type: String,
    pub pd: f64,
    pub lgd: f64,
    pub scenario_id: String,
    pub counterparty_code: String,
    pub company_code: String,
    pub deal_book: String,
    pub contract_reference: String,
    pub deal_reference: String,
    pub ead_source: String,
    pub valid_from: i64,
    pub valid_to: i64,
}

#[derive(Debug, Clone)]
pub struct CrEadOutput {
    pub loan_id: u64,
    pub ead: f64,
    pub ead_pre_ccf: f64,
    pub ead_principal: f64,
    pub ccf: f64,
    pub rw: f64,
    pub pd: f64,
    pub lgd: f64,
    pub product_kind: String,
    pub bs_type: String,
    pub rating: String,
    pub scenario_id: String,
    pub counterparty_code: String,
    pub company_code: String,
    pub deal_book: String,
    pub contract_reference: String,
    pub deal_reference: String,
    pub ead_formula_id: String,
    pub ccf_map_id: String,
    pub rw_map_id: String,
    pub method: String,
    pub diag: String,
    pub crm_pre_ccf_std: bool,
    pub valid_from: i64,
    pub valid_to: i64,
}

fn in_force(from: i64, to: i64, t: i64) -> bool {
    from <= t && t < to
}

fn match_star(cfg: &str, actual: &str) -> bool {
    let c = cfg.trim();
    if c.is_empty() || c == "*" {
        return true;
    }
    c.eq_ignore_ascii_case(actual.trim())
}

fn specificity(parts: &[&str]) -> i32 {
    parts
        .iter()
        .filter(|p| {
            let s = p.trim();
            !s.is_empty() && s != "*"
        })
        .count() as i32
}

fn pick_options<'a>(cfg: &'a CrEadConfig, supervisor: &str, t: i64) -> SupervisorOptions {
    let mut best: Option<&SupervisorOptions> = None;
    for o in &cfg.options {
        if !in_force(o.effective_from, o.effective_to, t) {
            continue;
        }
        if !match_star(&o.supervisor, supervisor) {
            continue;
        }
        best = Some(o);
        break;
    }
    best.cloned().unwrap_or_default()
}

fn pick_formula<'a>(
    formulas: &'a [EadFormulaRow],
    supervisor: &str,
    product_kind: &str,
    bs_type: &str,
    t: i64,
) -> Option<&'a EadFormulaRow> {
    let mut best: Option<(&EadFormulaRow, i32)> = None;
    for f in formulas {
        if !in_force(f.effective_from, f.effective_to, t) {
            continue;
        }
        if !match_star(&f.supervisor, supervisor)
            || !match_star(&f.product_kind, product_kind)
            || !match_star(&f.bs_type, bs_type)
        {
            continue;
        }
        let score = specificity(&[&f.product_kind, &f.bs_type, &f.book_type]);
        match best {
            None => best = Some((f, score)),
            Some((_, s)) if score > s => best = Some((f, score)),
            Some((_, s)) if score == s => {
                // keep first equally specific
            }
            _ => {}
        }
    }
    best.map(|(f, _)| f)
}

fn pick_ccf(
    maps: &[CcfMapRow],
    supervisor: &str,
    product_kind: &str,
    bs_type: &str,
    ccf_rating: &str,
    t: i64,
) -> (f64, String) {
    let mut best: Option<(&CcfMapRow, i32)> = None;
    for m in maps {
        if !in_force(m.effective_from, m.effective_to, t) {
            continue;
        }
        if !match_star(&m.supervisor, supervisor)
            || !match_star(&m.product_kind, product_kind)
            || !match_star(&m.bs_type, bs_type)
            || !match_star(&m.ccf_rating, ccf_rating)
        {
            continue;
        }
        let score = specificity(&[&m.product_kind, &m.bs_type, &m.ccf_rating]);
        match best {
            None => best = Some((m, score)),
            Some((_, s)) if score > s => best = Some((m, score)),
            _ => {}
        }
    }
    match best {
        Some((m, _)) => (m.ccf_value, m.map_id.clone()),
        None => (0.0, String::new()),
    }
}

fn pick_rw(
    maps: &[RwMapRow],
    supervisor: &str,
    approach: &str,
    product_kind: &str,
    rating: &str,
    t: i64,
) -> (f64, String) {
    let mut best: Option<(&RwMapRow, i32)> = None;
    for m in maps {
        if !in_force(m.effective_from, m.effective_to, t) {
            continue;
        }
        if !match_star(&m.supervisor, supervisor)
            || !match_star(&m.approach, approach)
            || !match_star(&m.product_kind, product_kind)
            || !match_star(&m.rating, rating)
        {
            continue;
        }
        let score = specificity(&[&m.product_kind, &m.rating, &m.claim_type]);
        match best {
            None => best = Some((m, score)),
            Some((_, s)) if score > s => best = Some((m, score)),
            _ => {}
        }
    }
    match best {
        Some((m, _)) => (m.rw, m.map_id.clone()),
        None => (1.0, String::new()),
    }
}

/// Compute one row; `as_of` must already be in-range for `exp` (caller filters).
pub fn compute_ead_rw(
    exp: &ExposureInput,
    cfg: &CrEadConfig,
    as_of: i64,
    supervisor: &str,
) -> CrEadOutput {
    let opts = pick_options(cfg, supervisor, as_of);
    let approach = if opts.approach.is_empty() {
        "STD"
    } else {
        opts.approach.as_str()
    };
    let mut diag: Vec<&str> = Vec::new();

    let bs_type = if exp.bs_type.trim().is_empty() {
        "ASS"
    } else {
        exp.bs_type.as_str()
    };

    let (ccf, ccf_map_id) = if let Some(c) = exp.ccf {
        if opts.ccf_explicit_wins {
            (c, "EXPLICIT".to_string())
        } else {
            let rating = if exp.ccf_rating.is_empty() {
                exp.rating.as_str()
            } else {
                exp.ccf_rating.as_str()
            };
            pick_ccf(
                &cfg.ccf_maps,
                supervisor,
                &exp.product_kind,
                bs_type,
                rating,
                as_of,
            )
        }
    } else {
        let rating = if exp.ccf_rating.is_empty() {
            if exp.rating.is_empty() {
                "*"
            } else {
                exp.rating.as_str()
            }
        } else {
            exp.ccf_rating.as_str()
        };
        let (v, id) = pick_ccf(
            &cfg.ccf_maps,
            supervisor,
            &exp.product_kind,
            bs_type,
            rating,
            as_of,
        );
        if id.is_empty() {
            diag.push("CCF_DEFAULT_0");
        }
        (v, id)
    };

    let formula = pick_formula(
        &cfg.formulas,
        supervisor,
        &exp.product_kind,
        bs_type,
        as_of,
    );
    let (mut expr, mut formula_id) = match formula {
        Some(f) => (f.expr_kind, f.formula_id.clone()),
        None => {
            diag.push("FORMULA_FALLBACK");
            (ExprKind::Outstanding, "FALLBACK_OUTSTANDING".into())
        }
    };

    let ead_source = exp.ead_source.trim().to_ascii_lowercase();
    if ead_source == "import" || (expr == ExprKind::ImportEpe && exp.epe > 0.0) {
        expr = ExprKind::ImportEpe;
        if ead_source == "import" {
            formula_id = "IMPORT".into();
        }
    }

    let base = if exp.outstanding > 0.0 {
        exp.outstanding
    } else {
        exp.ead
    };

    let (ead, ead_pre, ead_prin) = match expr {
        ExprKind::ImportEpe => {
            let e = if exp.epe > 0.0 { exp.epe } else { base };
            if exp.epe <= 0.0 {
                diag.push("IMPORT_EPE_EMPTY");
            }
            (e, e, if exp.outstanding > 0.0 { exp.outstanding } else { e })
        }
        ExprKind::OutPlusCcfUndrawn => {
            let ead = base + ccf * exp.undrawn;
            let ead_pre = base + exp.undrawn;
            (ead, ead_pre, base)
        }
        ExprKind::PrincipalOnly => (base, base + exp.undrawn, base),
        ExprKind::Outstanding => (base, base, base),
    };

    let (rw, rw_map_id) = pick_rw(
        &cfg.rw_maps,
        supervisor,
        approach,
        &exp.product_kind,
        &exp.rating,
        as_of,
    );
    let (rw, rw_map_id) = if rw_map_id.is_empty() {
        diag.push("RW_FALLBACK_1");
        (1.0, rw_map_id)
    } else {
        (rw, rw_map_id)
    };

    CrEadOutput {
        loan_id: exp.loan_id,
        ead,
        ead_pre_ccf: ead_pre,
        ead_principal: ead_prin,
        ccf,
        rw,
        pd: exp.pd,
        lgd: exp.lgd,
        product_kind: exp.product_kind.clone(),
        bs_type: bs_type.to_string(),
        rating: exp.rating.clone(),
        scenario_id: exp.scenario_id.clone(),
        counterparty_code: exp.counterparty_code.clone(),
        company_code: exp.company_code.clone(),
        deal_book: exp.deal_book.clone(),
        contract_reference: exp.contract_reference.clone(),
        deal_reference: exp.deal_reference.clone(),
        ead_formula_id: formula_id,
        ccf_map_id,
        rw_map_id,
        method: approach.to_string(),
        diag: diag.join(";"),
        crm_pre_ccf_std: opts.crm_pre_ccf_std,
        valid_from: exp.valid_from,
        valid_to: exp.valid_to,
    }
}

/// Filter exposures by as-of and compute all rows.
pub fn cr_ead_rw_batch(
    exposures: &[ExposureInput],
    cfg: &CrEadConfig,
    as_of: i64,
    supervisor: &str,
) -> Vec<CrEadOutput> {
    let mut out = Vec::new();
    for exp in exposures {
        if !in_force(exp.valid_from, exp.valid_to, as_of) {
            continue;
        }
        out.push(compute_ead_rw(exp, cfg, as_of, supervisor));
    }
    out.sort_by(|a, b| {
        a.loan_id
            .cmp(&b.loan_id)
            .then_with(|| a.deal_reference.cmp(&b.deal_reference))
            .then_with(|| a.valid_from.cmp(&b.valid_from))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray8_compat() -> CrEadConfig {
        CrEadConfig {
            formulas: vec![
                EadFormulaRow {
                    formula_id: "EAD_ASS_STD".into(),
                    supervisor: "HKMA".into(),
                    product_kind: "*".into(),
                    bs_type: "ASS".into(),
                    book_type: "*".into(),
                    expr_kind: ExprKind::OutPlusCcfUndrawn,
                    effective_from: 0,
                    effective_to: i64::MAX,
                },
                EadFormulaRow {
                    formula_id: "EAD_OFB_STD".into(),
                    supervisor: "HKMA".into(),
                    product_kind: "*".into(),
                    bs_type: "OFB".into(),
                    book_type: "*".into(),
                    expr_kind: ExprKind::OutPlusCcfUndrawn,
                    effective_from: 0,
                    effective_to: i64::MAX,
                },
            ],
            ccf_maps: vec![
                CcfMapRow {
                    map_id: "CCF_ASS".into(),
                    supervisor: "HKMA".into(),
                    ccf_rating: "*".into(),
                    ccf_value: 0.0,
                    product_kind: "*".into(),
                    bs_type: "ASS".into(),
                    effective_from: 0,
                    effective_to: i64::MAX,
                },
                CcfMapRow {
                    map_id: "CCF_STD".into(),
                    supervisor: "HKMA".into(),
                    ccf_rating: "A".into(),
                    ccf_value: 0.20,
                    product_kind: "*".into(),
                    bs_type: "OFB".into(),
                    effective_from: 0,
                    effective_to: i64::MAX,
                },
            ],
            rw_maps: vec![
                RwMapRow {
                    map_id: "RW_STD".into(),
                    supervisor: "HKMA".into(),
                    approach: "STD".into(),
                    claim_type: "*".into(),
                    rating: "BBB".into(),
                    product_kind: "*".into(),
                    rw: 1.0,
                    effective_from: 0,
                    effective_to: i64::MAX,
                },
                RwMapRow {
                    map_id: "RW_STD".into(),
                    supervisor: "HKMA".into(),
                    approach: "STD".into(),
                    claim_type: "*".into(),
                    rating: "A".into(),
                    product_kind: "*".into(),
                    rw: 0.5,
                    effective_from: 0,
                    effective_to: i64::MAX,
                },
            ],
            options: vec![SupervisorOptions {
                supervisor: "HKMA".into(),
                crm_pre_ccf_std: true,
                ..SupervisorOptions::default()
            }],
        }
    }

    #[test]
    fn ass_out_plus_ccf_undrawn() {
        let cfg = ray8_compat();
        let exp = ExposureInput {
            loan_id: 1,
            outstanding: 100_000.0,
            undrawn: 20_000.0,
            rating: "BBB".into(),
            product_kind: "loan_depo".into(),
            bs_type: "ASS".into(),
            valid_from: 0,
            valid_to: i64::MAX,
            ..Default::default()
        };
        let o = compute_ead_rw(&exp, &cfg, 0, "HKMA");
        assert!((o.ead - 100_000.0).abs() < 1e-9);
        assert!((o.ead_pre_ccf - 120_000.0).abs() < 1e-9);
        assert!((o.ccf - 0.0).abs() < 1e-9);
        assert!((o.rw - 1.0).abs() < 1e-9);
        assert_eq!(o.ead_formula_id, "EAD_ASS_STD");
        assert!(o.crm_pre_ccf_std);
    }

    #[test]
    fn ofb_ccf_a() {
        let cfg = ray8_compat();
        let exp = ExposureInput {
            loan_id: 2,
            outstanding: 0.0,
            undrawn: 100_000.0,
            rating: "A".into(),
            product_kind: "loan_depo".into(),
            bs_type: "OFB".into(),
            valid_from: 0,
            valid_to: i64::MAX,
            ..Default::default()
        };
        let o = compute_ead_rw(&exp, &cfg, 0, "HKMA");
        assert!((o.ccf - 0.20).abs() < 1e-9);
        assert!((o.ead - 20_000.0).abs() < 1e-9);
        assert!((o.ead_pre_ccf - 100_000.0).abs() < 1e-9);
        assert!((o.rw - 0.5).abs() < 1e-9);
    }

    #[test]
    fn import_epe() {
        let cfg = ray8_compat();
        let exp = ExposureInput {
            loan_id: 3,
            outstanding: 50_000.0,
            epe: 80_000.0,
            ead_source: "import".into(),
            rating: "BBB".into(),
            bs_type: "ASS".into(),
            valid_from: 0,
            valid_to: i64::MAX,
            ..Default::default()
        };
        let o = compute_ead_rw(&exp, &cfg, 0, "HKMA");
        assert_eq!(o.ead_formula_id, "IMPORT");
        assert!((o.ead - 80_000.0).abs() < 1e-9);
    }

    #[test]
    fn as_of_filters() {
        let cfg = ray8_compat();
        let rows = vec![
            ExposureInput {
                loan_id: 1,
                outstanding: 100.0,
                rating: "BBB".into(),
                bs_type: "ASS".into(),
                valid_from: 0,
                valid_to: 1000,
                deal_reference: "v0".into(),
                ..Default::default()
            },
            ExposureInput {
                loan_id: 1,
                outstanding: 80.0,
                rating: "BBB".into(),
                bs_type: "ASS".into(),
                valid_from: 1000,
                valid_to: i64::MAX,
                deal_reference: "v1".into(),
                ..Default::default()
            },
        ];
        let t0 = cr_ead_rw_batch(&rows, &cfg, 0, "HKMA");
        assert_eq!(t0.len(), 1);
        assert!((t0[0].ead - 100.0).abs() < 1e-9);
        let t1 = cr_ead_rw_batch(&rows, &cfg, 1000, "HKMA");
        assert_eq!(t1.len(), 1);
        assert!((t1[0].ead - 80.0).abs() < 1e-9);
    }
}
