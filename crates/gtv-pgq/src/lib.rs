//! gtv-pgq: SQL/PGQ (ISO/IEC 9075-16:2023) **L1 subset** rewritten onto the
//! engine's existing relations.
//!
//! # What this crate is
//!
//! A front-end rewrite, not an execution engine. `GRAPH_TABLE (g MATCH … COLUMNS …)`
//! is parsed, resolved against a [`PropertyGraphRegistry`], and replaced with an
//! ordinary `SELECT … FROM … JOIN …` over tables the caller already registered.
//! DataFusion plans that join with its normal CBO, so:
//!
//! * `EXPLAIN` shows the base relations, not a black box;
//! * there is no second planner, no forked `sqlparser`, no new operator;
//! * registering `gtv_batch_neighbors` / a GTVSeg edge table as the edge
//!   relation makes PGQ compose with the graph operators for free.
//!
//! # What L1 covers (see `docs/sql-pgq-conformance.md`)
//!
//! | feature | status |
//! |---|---|
//! | single path pattern, fixed length 1..=3 | ✅ |
//! | direction `->` / `<-` | ✅ |
//! | one label per node / edge | ✅ (multiple labels: ❌) |
//! | inline `WHERE` on node / edge, outer `WHERE` | ✅ |
//! | `COLUMNS (expr AS name, …)` | ✅ |
//! | bounded quantifier `{1,n}`, same-label endpoints | △ |
//! | `ANY SHORTEST`, `ALL SHORTEST`, path modes | ❌ (L2) |
//! | unbounded `*` / `+` | ❌ (L2) |
//! | `MATCH` with multiple comma-separated patterns | ❌ (L2) |
//!
//! # What it refuses to do
//!
//! Anything outside the table above is a [`PgqError::Unsupported`] naming the
//! feature, never a silently different query. That is the whole point of
//! publishing a conformance table: an unsupported query must fail loudly.
//!
//! # Example
//!
//! ```
//! use gtv_pgq::{PropertyGraphRegistry, rewrite};
//!
//! let mut reg = PropertyGraphRegistry::new();
//! reg.add_graph("g");
//! reg.add_vertex("g", "Account", "accounts", "id").unwrap();
//! reg.add_edge("g", "Transfer", "transfers", "src_id", "dst_id").unwrap();
//!
//! let sql = "SELECT t.src, t.dst FROM GRAPH_TABLE (g \
//!            MATCH (a:Account)-[e:Transfer]->(b:Account) \
//!            COLUMNS (a.id AS src, b.id AS dst)) AS t";
//! let out = rewrite(sql, &reg).unwrap();
//! assert!(out.sql.contains("FROM transfers"));
//! assert!(!out.sql.contains("GRAPH_TABLE"));
//! ```

pub mod lexer;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use lexer::{ident_value, is_keyword, matching_paren, significant, split_top_level_commas, Kind, Token};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PgqError {
    #[error("statement contains no GRAPH_TABLE clause")]
    NoGraphTable,
    #[error("unbalanced parentheses in GRAPH_TABLE clause")]
    UnbalancedParens,
    #[error("unknown property graph `{0}`")]
    UnknownGraph(String),
    #[error("unknown vertex label `{label}` in graph `{graph}`")]
    UnknownVertexLabel { graph: String, label: String },
    #[error("unknown edge label `{label}` in graph `{graph}`")]
    UnknownEdgeLabel { graph: String, label: String },
    #[error("parse error in GRAPH_TABLE: expected {expected}, found `{found}`")]
    Parse { expected: String, found: String },
    #[error("unsupported SQL/PGQ feature: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, PgqError>;

fn parse_err(expected: impl Into<String>, found: &str) -> PgqError {
    PgqError::Parse {
        expected: expected.into(),
        found: found.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// A vertex table backing one label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexDef {
    pub label: String,
    pub table: String,
    /// Primary-key column; a bare variable reference in `COLUMNS` resolves to it.
    pub key: String,
}

/// An edge table backing one label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeDef {
    pub label: String,
    pub table: String,
    pub src: String,
    pub dst: String,
    /// When set, the rewrite adds `edge_type = N` so a single physical edge
    /// table can carry several PGQ labels.
    pub edge_type: Option<u16>,
}

/// One named property graph: labels resolved to physical relations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphDef {
    pub name: String,
    pub vertices: BTreeMap<String, VertexDef>,
    pub edges: BTreeMap<String, EdgeDef>,
}

impl GraphDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// The only vertex label, when there is exactly one — lets `(a)` omit `:Label`.
    fn sole_vertex_label(&self) -> Option<&str> {
        if self.vertices.len() == 1 {
            self.vertices.keys().next().map(String::as_str)
        } else {
            None
        }
    }
}

/// Property graphs known to this session.
///
/// Kept **out** of the DataFusion catalog on purpose: `CREATE PROPERTY GRAPH`
/// in PostgreSQL 19 was reverted because of how it interacted with catalog
/// representation, and we only need name → (table, key) resolution at rewrite
/// time. A separate registry keeps that concern separate.
#[derive(Debug, Clone, Default)]
pub struct PropertyGraphRegistry {
    graphs: BTreeMap<String, GraphDef>,
}

impl PropertyGraphRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_graph(&mut self, name: impl Into<String>) -> &mut GraphDef {
        let name = name.into();
        self.graphs
            .entry(name.clone())
            .or_insert_with(|| GraphDef::new(name))
    }

    pub fn add_vertex(
        &mut self,
        graph: &str,
        label: &str,
        table: &str,
        key: &str,
    ) -> Result<&mut Self> {
        let g = self
            .graphs
            .get_mut(graph)
            .ok_or_else(|| PgqError::UnknownGraph(graph.to_string()))?;
        g.vertices.insert(
            label.to_string(),
            VertexDef {
                label: label.to_string(),
                table: table.to_string(),
                key: key.to_string(),
            },
        );
        Ok(self)
    }

    pub fn add_edge(
        &mut self,
        graph: &str,
        label: &str,
        table: &str,
        src: &str,
        dst: &str,
    ) -> Result<&mut Self> {
        let g = self
            .graphs
            .get_mut(graph)
            .ok_or_else(|| PgqError::UnknownGraph(graph.to_string()))?;
        g.edges.insert(
            label.to_string(),
            EdgeDef {
                label: label.to_string(),
                table: table.to_string(),
                src: src.to_string(),
                dst: dst.to_string(),
                edge_type: None,
            },
        );
        Ok(self)
    }

    /// Tag an edge label with the numeric type stored in the physical table.
    pub fn set_edge_type(&mut self, graph: &str, label: &str, edge_type: u16) -> Result<&mut Self> {
        let g = self
            .graphs
            .get_mut(graph)
            .ok_or_else(|| PgqError::UnknownGraph(graph.to_string()))?;
        let edge = g
            .edges
            .get_mut(label)
            .ok_or_else(|| PgqError::UnknownEdgeLabel {
                graph: graph.to_string(),
                label: label.to_string(),
            })?;
        edge.edge_type = Some(edge_type);
        Ok(self)
    }

    pub fn graph(&self, name: &str) -> Option<&GraphDef> {
        self.graphs.get(name)
    }

    pub fn graph_names(&self) -> impl Iterator<Item = &str> {
        self.graphs.keys().map(String::as_str)
    }
}

// ---------------------------------------------------------------------------
// Rewrite
// ---------------------------------------------------------------------------

/// Result of rewriting one statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rewrite {
    /// Statement with every `GRAPH_TABLE ( … )` replaced by a plain `SELECT`.
    pub sql: String,
    /// Physical relations the rewrite reads, in first-use order.
    pub relations: Vec<String>,
    /// Number of `GRAPH_TABLE` clauses rewritten.
    pub clauses: usize,
}

/// Rewrite every `GRAPH_TABLE` clause in `sql`.
///
/// Returns [`PgqError::NoGraphTable`] when there is nothing to do; callers that
/// want a pass-through should check [`contains_graph_table`] first.
pub fn rewrite(sql: &str, registry: &PropertyGraphRegistry) -> Result<Rewrite> {
    let tokens = significant(sql);
    let mut out = String::with_capacity(sql.len() + 256);
    let mut cursor = 0usize;
    let mut clauses = 0usize;
    let mut relations: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if !is_keyword(sql, &tokens[i], "GRAPH_TABLE") {
            i += 1;
            continue;
        }
        // The next significant token must be `(`.
        let open = i + 1;
        if tokens.get(open).map(|t| t.text(sql)) != Some("(") {
            return Err(parse_err("`(` after GRAPH_TABLE", tokens.get(open).map(|t| t.text(sql)).unwrap_or("<eof>")));
        }
        let close = matching_paren(sql, &tokens, open).ok_or(PgqError::UnbalancedParens)?;

        let body = &sql[tokens[open].end..tokens[close].start];
        let plan = plan_clause(body, registry)?;
        relations.extend(plan.relations.iter().cloned());

        out.push_str(&sql[cursor..tokens[i].start]);
        out.push('(');
        out.push_str(&plan.sql);
        out.push(')');
        cursor = tokens[close].end;
        i = close + 1;
        clauses += 1;
    }

    if clauses == 0 {
        return Err(PgqError::NoGraphTable);
    }
    out.push_str(&sql[cursor..]);
    Ok(Rewrite {
        sql: out,
        relations,
        clauses,
    })
}

/// Cheap check so a caller can skip the rewrite for plain SQL.
pub fn contains_graph_table(sql: &str) -> bool {
    significant(sql)
        .iter()
        .any(|t| is_keyword(sql, t, "GRAPH_TABLE"))
}

// ---------------------------------------------------------------------------
// Clause parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct NodePat {
    var: String,
    label: Option<String>,
    predicate: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Forward,
    Backward,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RelPat {
    var: Option<String>,
    label: Option<String>,
    predicate: Option<String>,
    direction: Direction,
    min: usize,
    max: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ColumnItem {
    expr: String,
    alias: String,
}

#[derive(Debug, Clone)]
struct Clause {
    graph: String,
    nodes: Vec<NodePat>,
    rels: Vec<RelPat>,
    predicate: Option<String>,
    columns: Vec<ColumnItem>,
}

struct Plan {
    sql: String,
    relations: Vec<String>,
}

/// State for the tiny recursive-descent parser over the clause body.
struct Parser<'a> {
    src: &'a str,
    toks: Vec<Token>,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            toks: significant(src),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.pos)
    }

    fn text_at(&self, i: usize) -> &'a str {
        self.toks.get(i).map(|t| t.text(self.src)).unwrap_or("")
    }

    fn eat_punct(&mut self, p: &str) -> bool {
        if self.peek().map(|t| t.text(self.src)) == Some(p) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn eat_keyword(&mut self, kw: &str) -> bool {
        if self.peek().map(|t| is_keyword(self.src, t, kw)).unwrap_or(false) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_punct(&mut self, p: &str) -> Result<()> {
        if self.eat_punct(p) {
            Ok(())
        } else {
            Err(parse_err(
                format!("`{p}`"),
                self.peek().map(|t| t.text(self.src)).unwrap_or("<eof>"),
            ))
        }
    }

    fn expect_ident(&mut self, what: &str) -> Result<String> {
        match self.peek() {
            Some(t) if matches!(t.kind, Kind::Ident | Kind::QuotedIdent) => {
                let v = ident_value(self.src, t);
                self.pos += 1;
                Ok(v)
            }
            Some(t) => Err(parse_err(what, t.text(self.src))),
            None => Err(parse_err(what, "<eof>")),
        }
    }

    /// Predicate text after `WHERE`, up to the element's closing bracket.
    ///
    /// `open` is the index of the element's opening `(` / `[`. `self.pos` must
    /// already point past `WHERE`; on return it points past the closer.
    fn take_predicate_until(&mut self, open: usize) -> Result<String> {
        let close = lexer::matching(self.src, &self.toks, open).ok_or(PgqError::UnbalancedParens)?;
        if self.pos >= close {
            return Err(parse_err("a predicate", "<empty>"));
        }
        let start = self.toks[self.pos - 1].end;
        let text = self.src[start..self.toks[close].start].trim().to_string();
        self.pos = close + 1;
        Ok(text)
    }

    /// `->`, or `-` `>` split into two tokens.
    fn eat_arrow_forward(&mut self) -> bool {
        if self.peek().map(|t| t.text(self.src)) == Some("->") {
            self.pos += 1;
            return true;
        }
        if self.peek().map(|t| t.text(self.src)) == Some("-")
            && self
                .toks
                .get(self.pos + 1)
                .map(|t| t.text(self.src))
                == Some(">")
        {
            self.pos += 2;
            return true;
        }
        false
    }

    /// `<-`, or `<` `-` split into two tokens.
    fn eat_arrow_backward(&mut self) -> bool {
        if self.peek().map(|t| t.text(self.src)) == Some("<-") {
            self.pos += 1;
            return true;
        }
        if self.peek().map(|t| t.text(self.src)) == Some("<")
            && self
                .toks
                .get(self.pos + 1)
                .map(|t| t.text(self.src))
                == Some("-")
        {
            self.pos += 2;
            return true;
        }
        false
    }

    fn parse_clause(&mut self) -> Result<Clause> {
        let graph = self.expect_ident("property graph name")?;
        if !self.eat_keyword("MATCH") {
            return Err(parse_err("MATCH", self.text_at(self.pos)));
        }
        let (nodes, rels) = self.parse_pattern()?;

        let predicate = if self.eat_keyword("WHERE") {
            Some(self.parse_raw_predicate(&["COLUMNS", "COLUMNS"]) ?)
        } else {
            None
        };

        if !self.eat_keyword("COLUMNS") {
            return Err(parse_err("COLUMNS", self.text_at(self.pos)));
        }
        let columns = self.parse_columns()?;

        if let Some(t) = self.peek() {
            return Err(PgqError::Unsupported(format!(
                "trailing input in GRAPH_TABLE after COLUMNS: `{}`",
                t.text(self.src)
            )));
        }
        Ok(Clause {
            graph,
            nodes,
            rels,
            predicate,
            columns,
        })
    }

    /// Everything up to one of `stops` at depth 0.
    fn parse_raw_predicate(&mut self, stops: &[&str]) -> Result<String> {
        let start = self.pos;
        let mut depth = 0usize;
        while let Some(t) = self.peek() {
            let text = t.text(self.src);
            match text {
                "(" | "[" => depth += 1,
                ")" | "]" => depth = depth.saturating_sub(1),
                _ => {}
            }
            if depth == 0 && stops.iter().any(|s| is_keyword(self.src, t, s)) {
                break;
            }
            self.pos += 1;
        }
        if self.pos == start {
            return Err(parse_err("a predicate", "<end of clause>"));
        }
        let first = &self.toks[start];
        let last = &self.toks[self.pos - 1];
        Ok(self.src[first.start..last.end].trim().to_string())
    }

    fn parse_pattern(&mut self) -> Result<(Vec<NodePat>, Vec<RelPat>)> {
        let mut nodes = Vec::new();
        let mut rels = Vec::new();
        nodes.push(self.parse_node()?);
        loop {
            if self.peek().map(|t| t.text(self.src)) == Some(",") {
                return Err(PgqError::Unsupported(
                    "multiple comma-separated path patterns (L2)".into(),
                ));
            }
            let Some(rel) = self.try_parse_rel()? else {
                break;
            };
            rels.push(rel);
            nodes.push(self.parse_node()?);
        }
        if rels.is_empty() {
            return Err(PgqError::Unsupported(
                "node-only patterns (no relationship) are not part of L1".into(),
            ));
        }
        Ok((nodes, rels))
    }

    fn parse_node(&mut self) -> Result<NodePat> {
        let open = self.pos;
        self.expect_punct("(")?;
        let mut var = None;
        let mut label = None;
        if let Some(t) = self.peek() {
            if matches!(t.kind, Kind::Ident | Kind::QuotedIdent)
                && !is_keyword(self.src, t, "WHERE")
            {
                var = Some(self.expect_ident("variable")?);
                if self.eat_punct(":") {
                    label = Some(self.parse_label_list()?);
                }
            } else if t.text(self.src) == ":" {
                self.pos += 1;
                label = Some(self.parse_label_list()?);
            }
        }
        let predicate = if self.eat_keyword("WHERE") {
            Some(self.take_predicate_until(open)?)
        } else {
            self.expect_punct(")")?;
            None
        };
        Ok(NodePat {
            var: var.unwrap_or_default(),
            label,
            predicate,
        })
    }

    fn parse_label_list(&mut self) -> Result<String> {
        let first = self.expect_ident("label")?;
        if self.peek().map(|t| t.text(self.src)) == Some("|") {
            return Err(PgqError::Unsupported(
                "multiple labels on one element (`:A|B`) (L2)".into(),
            ));
        }
        Ok(first)
    }

    fn try_parse_rel(&mut self) -> Result<Option<RelPat>> {
        let saved = self.pos;
        let direction = if self.eat_arrow_backward() {
            Direction::Backward
        } else if self.eat_punct("-") {
            Direction::Forward
        } else {
            return Ok(None);
        };
        if self.peek().map(|t| t.text(self.src)) != Some("[") {
            self.pos = saved;
            return Ok(None);
        }

        let open = self.pos;
        self.expect_punct("[")?;
        let mut var = None;
        let mut label = None;
        if self.peek().map(|t| t.text(self.src)) == Some(":") {
            self.pos += 1;
            label = Some(self.parse_label_list()?);
        } else if self
            .peek()
            .map(|x| matches!(x.kind, Kind::Ident | Kind::QuotedIdent))
            .unwrap_or(false)
            && !is_keyword(self.src, self.peek().unwrap(), "WHERE")
        {
            var = Some(self.expect_ident("edge variable")?);
            if self.eat_punct(":") {
                label = Some(self.parse_label_list()?);
            }
        }
        let predicate = if self.eat_keyword("WHERE") {
            Some(self.take_predicate_until(open)?)
        } else {
            self.expect_punct("]")?;
            None
        };

        match direction {
            Direction::Forward => {
                if !self.eat_arrow_forward() {
                    return Err(parse_err("`->`", self.text_at(self.pos)));
                }
            }
            Direction::Backward => {
                self.expect_punct("-")?;
            }
        }

        let (min, max) = self.parse_quantifier()?;
        Ok(Some(RelPat {
            var,
            label,
            predicate,
            direction,
            min,
            max,
        }))
    }

    fn parse_quantifier(&mut self) -> Result<(usize, usize)> {
        if !self.eat_punct("{") {
            return Ok((1, 1));
        }
        let lo = self.expect_number()?;
        let (min, max) = if self.eat_punct(",") {
            let hi = self.expect_number()?;
            (lo, hi)
        } else {
            (lo, lo)
        };
        if !self.eat_punct("}") {
            return Err(parse_err("`}`", self.text_at(self.pos)));
        }
        if min == 0 {
            return Err(PgqError::Unsupported(
                "zero-length paths (`{0,n}`) (L2)".into(),
            ));
        }
        if max > 3 {
            return Err(PgqError::Unsupported(format!(
                "bounded path length above 3 (`{{1,{max}}}`); L1 caps at 3"
            )));
        }
        if min > max {
            return Err(parse_err("a quantifier with lower <= upper", "{m,n}"));
        }
        Ok((min, max))
    }

    fn expect_number(&mut self) -> Result<usize> {
        match self.peek() {
            Some(t) if t.kind == Kind::Number => {
                let v: usize = t.text(self.src).parse().map_err(|_| {
                    parse_err("an integer path length", t.text(self.src))
                })?;
                self.pos += 1;
                Ok(v)
            }
            Some(t) => Err(parse_err("an integer path length", t.text(self.src))),
            None => Err(parse_err("an integer path length", "<eof>")),
        }
    }

    fn parse_columns(&mut self) -> Result<Vec<ColumnItem>> {
        let open = self.pos;
        if self.peek().map(|t| t.text(self.src)) != Some("(") {
            return Err(parse_err("`(` after COLUMNS", self.text_at(self.pos)));
        }
        let close = matching_paren(self.src, &self.toks, open).ok_or(PgqError::UnbalancedParens)?;
        let inner = &self.toks[open + 1..close];
        let groups = split_top_level_commas(self.src, inner);
        if groups.is_empty() {
            return Err(parse_err("at least one COLUMNS item", "empty COLUMNS"));
        }
        let mut items = Vec::new();
        for (a, b) in groups {
            let group = &inner[a..b];
            // Find the last top-level `AS`.
            let mut depth = 0usize;
            let mut as_at = None;
            for (i, t) in group.iter().enumerate() {
                match t.text(self.src) {
                    "(" | "[" => depth += 1,
                    ")" | "]" => depth = depth.saturating_sub(1),
                    _ => {}
                }
                if depth == 0 && is_keyword(self.src, t, "AS") {
                    as_at = Some(i);
                }
            }
            let Some(i) = as_at else {
                return Err(parse_err(
                    "`expr AS name` in COLUMNS",
                    self.src[group[0].start..group[group.len() - 1].end].trim(),
                ));
            };
            let expr = self.src[group[0].start..group[i - 1].end].trim().to_string();
            let alias_tok = &group[i + 1];
            if !matches!(alias_tok.kind, Kind::Ident | Kind::QuotedIdent) {
                return Err(parse_err("a column alias", alias_tok.text(self.src)));
            }
            if group.len() != i + 2 {
                return Err(PgqError::Unsupported(
                    "expressions after a COLUMNS alias".into(),
                ));
            }
            items.push(ColumnItem {
                expr,
                alias: ident_value(self.src, alias_tok),
            });
        }
        self.pos = close + 1;
        Ok(items)
    }
}

fn plan_clause(body: &str, registry: &PropertyGraphRegistry) -> Result<Plan> {
    let mut parser = Parser::new(body);
    let clause = parser.parse_clause()?;
    build_plan(&clause, registry)
}

// ---------------------------------------------------------------------------
// Plan building
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct VarBinding {
    alias: String,
    key: String,
}

fn build_plan(clause: &Clause, registry: &PropertyGraphRegistry) -> Result<Plan> {
    let graph = registry
        .graph(&clause.graph)
        .ok_or_else(|| PgqError::UnknownGraph(clause.graph.clone()))?;
    if clause.rels.len() != 1 {
        return Err(PgqError::Unsupported(format!(
            "{} hops in one pattern; L1 supports exactly one relationship per pattern",
            clause.rels.len()
        )));
    }
    let rel = &clause.rels[0];
    if rel.min != 1 || rel.max < 1 {
        return Err(PgqError::Unsupported(
            "path quantifiers other than 1..n are L2".into(),
        ));
    }
    if rel.max > 3 {
        return Err(PgqError::Unsupported("path length above 3".into()));
    }

    let start = &clause.nodes[0];
    let end = &clause.nodes[1];
    let start_label = resolve_label(graph, start.label.as_deref(), true)?;
    let end_label = resolve_label(graph, end.label.as_deref(), true)?;
    let edge_label = match &rel.label {
        Some(l) => l.clone(),
        None => {
            if graph.edges.len() == 1 {
                graph.edges.keys().next().unwrap().clone()
            } else {
                return Err(PgqError::Unsupported(
                    "relationship without a label in a graph with several edge labels".into(),
                ));
            }
        }
    };
    let edge = graph
        .edges
        .get(&edge_label)
        .ok_or_else(|| PgqError::UnknownEdgeLabel {
            graph: graph.name.clone(),
            label: edge_label.clone(),
        })?;
    let v_start = &graph.vertices[&start_label];
    let v_end = &graph.vertices[&end_label];

    if rel.max > 1 && start_label != end_label {
        return Err(PgqError::Unsupported(format!(
            "variable-length path between different labels (`{start_label}` → `{end_label}`); \
             L1 requires both endpoints to share a label"
        )));
    }

    let mut relations: Vec<String> = Vec::new();
    let push_rel = |name: &str, out: &mut Vec<String>| {
        if !out.iter().any(|r| r == name) {
            out.push(name.to_string());
        }
    };
    push_rel(&edge.table, &mut relations);
    push_rel(&v_start.table, &mut relations);
    push_rel(&v_end.table, &mut relations);

    // One SELECT branch per allowed path length.
    let mut branches = Vec::new();
    for len in rel.min..=rel.max {
        let mut sql = String::new();
        let mut bindings: BTreeMap<String, VarBinding> = BTreeMap::new();
        let node_alias = |pat: &NodePat, fallback: &str| -> String {
            if pat.var.is_empty() {
                fallback.to_string()
            } else {
                pat.var.clone()
            }
        };

        let start_alias = node_alias(start, "v0");
        let end_alias = node_alias(end, &format!("v{len}"));

        // FROM edge 1
        let e1 = rel.var.clone().unwrap_or_else(|| "e1".to_string());
        write!(sql, "SELECT {{COLS}} FROM {} AS {}", edge.table, e1).unwrap();

        // Join start vertex to edge 1.
        let (e1_src, e1_dst) = match rel.direction {
            Direction::Forward => (&edge.src, &edge.dst),
            Direction::Backward => (&edge.dst, &edge.src),
        };
        write!(
            sql,
            " JOIN {} AS {} ON {}.{} = {}.{}",
            v_start.table, start_alias, start_alias, v_start.key, e1, e1_src
        )
        .unwrap();

        bindings.insert(
            start.var.clone(),
            VarBinding {
                alias: start_alias.clone(),
                key: v_start.key.clone(),
            },
        );
        if let Some(var) = &rel.var {
            bindings.insert(
                var.clone(),
                VarBinding {
                    alias: e1.clone(),
                    key: edge.src.clone(),
                },
            );
        }

        // Intermediate hops: chain edge tables, bind intermediate nodes to the
        // shared endpoint label (checked above).
        let mut prev_edge = e1.clone();
        for hop in 2..=len {
            let e = format!("e{hop}");
            let v = format!("v{hop}");
            write!(
                sql,
                " JOIN {} AS {} ON {}.{} = {}.{}",
                edge.table, e, e, edge.src, prev_edge, e1_dst
            )
            .unwrap();
            write!(
                sql,
                " JOIN {} AS {} ON {}.{} = {}.{}",
                v_start.table, v, v, v_start.key, e, edge.src
            )
            .unwrap();
            prev_edge = e;
        }

        // Join the end vertex.
        write!(
            sql,
            " JOIN {} AS {} ON {}.{} = {}.{}",
            v_end.table, end_alias, end_alias, v_end.key, prev_edge, e1_dst
        )
        .unwrap();
        bindings.insert(
            end.var.clone(),
            VarBinding {
                alias: end_alias,
                key: v_end.key.clone(),
            },
        );

        // Predicates.
        let mut conjuncts: Vec<String> = Vec::new();
        if let Some(n) = &start.predicate {
            conjuncts.push(substitute(n, &bindings));
        }
        if let Some(n) = &end.predicate {
            conjuncts.push(substitute(n, &bindings));
        }
        if let Some(p) = &rel.predicate {
            // The relationship predicate binds to the first edge; for a longer
            // path it applies to every hop, matching PGQ's per-edge semantics.
            let mut edges = vec![e1.clone()];
            for hop in 2..=len {
                edges.push(format!("e{hop}"));
            }
            for e in edges {
                let mut b = bindings.clone();
                b.insert(
                    rel.var.clone().unwrap_or_else(|| "e".to_string()),
                    VarBinding {
                        alias: e,
                        key: edge.src.clone(),
                    },
                );
                conjuncts.push(substitute(p, &b));
            }
        }
        if let Some(p) = &edge.edge_type {
            conjuncts.push(format!("{e1}.edge_type = {p}"));
            for hop in 2..=len {
                conjuncts.push(format!("e{hop}.edge_type = {p}"));
            }
        }
        if let Some(p) = &clause.predicate {
            conjuncts.push(substitute(p, &bindings));
        }
        if !conjuncts.is_empty() {
            write!(sql, " WHERE {}", conjuncts.join(" AND ")).unwrap();
        }

        // Columns.
        let cols: Vec<String> = clause
            .columns
            .iter()
            .map(|c| {
                let expr = substitute(&c.expr, &bindings);
                format!("{expr} AS {}", quote_if_needed(&c.alias))
            })
            .collect();
        branches.push(sql.replace("{COLS}", &cols.join(", ")));
    }

    let sql = if branches.len() == 1 {
        branches.pop().unwrap()
    } else {
        branches
            .into_iter()
            .map(|b| format!("({b})"))
            .collect::<Vec<_>>()
            .join(" UNION ALL ")
    };

    Ok(Plan { sql, relations })
}

fn resolve_label(graph: &GraphDef, label: Option<&str>, vertex: bool) -> Result<String> {
    match label {
        Some(l) => {
            let ok = if vertex {
                graph.vertices.contains_key(l)
            } else {
                graph.edges.contains_key(l)
            };
            if ok {
                Ok(l.to_string())
            } else if vertex {
                Err(PgqError::UnknownVertexLabel {
                    graph: graph.name.clone(),
                    label: l.to_string(),
                })
            } else {
                Err(PgqError::UnknownEdgeLabel {
                    graph: graph.name.clone(),
                    label: l.to_string(),
                })
            }
        }
        None => graph.sole_vertex_label().map(str::to_string).ok_or_else(|| {
            PgqError::Unsupported(format!(
                "node without a label in graph `{}`, which has {} vertex labels",
                graph.name,
                graph.vertices.len()
            ))
        }),
    }
}

/// Replace `var` / `var.property` references with their SQL alias.
///
/// Walks the original text so whitespace, string literals and comments are
/// preserved byte-for-byte; only identifier tokens that name a pattern variable
/// are touched. A bare variable becomes `alias.key` (the vertex key), which is
/// what `COLUMNS (a AS id)` means in SQL/PGQ.
fn substitute(fragment: &str, bindings: &BTreeMap<String, VarBinding>) -> String {
    let toks = lexer::lex(fragment);
    let mut out = String::with_capacity(fragment.len() + 16);
    for (i, t) in toks.iter().enumerate() {
        if t.kind == Kind::Ident {
            let name = t.text(fragment);
            if let Some(b) = bindings.get(name) {
                // Look ahead for `.` to decide between `alias` and `alias.key`.
                let next = toks[i + 1..].iter().find(|x| x.kind != Kind::Space);
                if next.map(|x| x.text(fragment)) == Some(".") {
                    out.push_str(&b.alias);
                } else {
                    out.push_str(&b.alias);
                    out.push('.');
                    out.push_str(&b.key);
                }
                continue;
            }
        }
        out.push_str(t.text(fragment));
    }
    out
}

fn quote_if_needed(name: &str) -> String {
    let simple = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.chars().next().unwrap().is_ascii_digit();
    if simple {
        name.to_string()
    } else {
        format!("\"{}\"", name.replace('"', "\"\""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> PropertyGraphRegistry {
        let mut reg = PropertyGraphRegistry::new();
        reg.add_graph("g");
        reg.add_vertex("g", "Account", "accounts", "id").unwrap();
        reg.add_vertex("g", "Customer", "customers", "cid").unwrap();
        reg.add_edge("g", "Transfer", "transfers", "src_id", "dst_id")
            .unwrap();
        reg
    }

    #[test]
    fn one_hop_rewrites_to_a_join() {
        let sql = "SELECT t.a, t.b FROM GRAPH_TABLE (g \
                   MATCH (a:Account)-[e:Transfer]->(b:Account) \
                   COLUMNS (a.id AS a, b.id AS b)) AS t";
        let out = rewrite(sql, &registry()).unwrap();
        assert!(out.sql.contains("FROM transfers AS e"));
        assert!(out.sql.contains("JOIN accounts AS a ON a.id = e.src_id"));
        assert!(out.sql.contains("JOIN accounts AS b ON b.id = e.dst_id"));
        assert!(out.sql.contains("a.id AS a, b.id AS b"));
        assert!(!out.sql.contains("GRAPH_TABLE"));
        assert_eq!(out.relations, vec!["transfers", "accounts"]);
    }

    #[test]
    fn backward_edge_swaps_endpoints() {
        let sql = "SELECT * FROM GRAPH_TABLE (g \
                   MATCH (a:Account)<-[e:Transfer]-(b:Account) \
                   COLUMNS (a.id AS a)) AS t";
        let out = rewrite(sql, &registry()).unwrap();
        assert!(out.sql.contains("a.id = e.dst_id"));
        assert!(out.sql.contains("b.id = e.src_id"));
    }

    #[test]
    fn predicates_and_columns_are_substituted() {
        let sql = "SELECT * FROM GRAPH_TABLE (g \
                   MATCH (a:Account WHERE a.branch = 'HK')-[e:Transfer WHERE e.amount > 100]->(b:Account) \
                   WHERE b.status = 'open' \
                   COLUMNS (a.id AS src, e.amount AS amt)) AS t";
        let out = rewrite(sql, &registry()).unwrap();
        assert!(out.sql.contains("a.branch = 'HK'"));
        assert!(out.sql.contains("e.amount > 100"));
        assert!(out.sql.contains("b.status = 'open'"));
        assert!(out.sql.contains("e.amount AS amt"));
    }

    #[test]
    fn bare_variable_becomes_the_key() {
        let sql = "SELECT * FROM GRAPH_TABLE (g \
                   MATCH (a:Account)-[e:Transfer]->(b:Account) COLUMNS (a AS src)) AS t";
        let out = rewrite(sql, &registry()).unwrap();
        assert!(out.sql.contains("a.id AS src"));
    }

    #[test]
    fn quantifier_expands_to_a_union() {
        let sql = "SELECT * FROM GRAPH_TABLE (g \
                   MATCH (a:Account)-[e:Transfer]->{1,2}(b:Account) \
                   COLUMNS (a.id AS src, b.id AS dst)) AS t";
        let out = rewrite(sql, &registry()).unwrap();
        assert_eq!(out.sql.matches("UNION ALL").count(), 1);
        assert!(out.sql.contains("e2.src_id"), "second hop must chain on src");
    }

    #[test]
    fn quantifier_across_labels_is_refused() {
        let sql = "SELECT * FROM GRAPH_TABLE (g \
                   MATCH (a:Account)-[e:Transfer]->{1,2}(b:Customer) \
                   COLUMNS (a.id AS src)) AS t";
        let err = rewrite(sql, &registry()).unwrap_err();
        assert!(matches!(err, PgqError::Unsupported(_)));
        assert!(err.to_string().contains("variable-length"));
    }

    #[test]
    fn unknown_label_is_named_in_the_error() {
        let sql = "SELECT * FROM GRAPH_TABLE (g \
                   MATCH (a:NoSuch)-[e:Transfer]->(b:Account) COLUMNS (a.id AS x)) AS t";
        let err = rewrite(sql, &registry()).unwrap_err();
        assert!(matches!(err, PgqError::UnknownVertexLabel { .. }));
        assert!(err.to_string().contains("NoSuch"));
    }

    #[test]
    fn no_graph_table_is_reported() {
        assert_eq!(
            rewrite("SELECT 1", &registry()).unwrap_err(),
            PgqError::NoGraphTable
        );
        assert!(!contains_graph_table("SELECT 1"));
        assert!(contains_graph_table("select * from graph_table (g match ..."));
    }

    #[test]
    fn edge_type_filter_is_added() {
        let mut reg = registry();
        reg.set_edge_type("g", "Transfer", 3).unwrap();
        let sql = "SELECT * FROM GRAPH_TABLE (g \
                   MATCH (a:Account)-[e:Transfer]->(b:Account) COLUMNS (a.id AS x)) AS t";
        let out = rewrite(sql, &reg).unwrap();
        assert!(out.sql.contains("e.edge_type = 3"));
    }
}
