//! Lexer for the SQL/PGQ subset.
//!
//! Deliberately not a general SQL lexer: it only has to (a) find balanced
//! parentheses in an outer statement, (b) tokenise a `GRAPH_TABLE ( … )` body,
//! and (c) rewrite `var.property` references inside raw `WHERE` / `COLUMNS`
//! fragments without touching string literals or comments.

/// What a token is. Only the distinctions the rewriter cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Ident,
    QuotedIdent,
    Number,
    String,
    /// Two-character operator such as `->`, `<-`, `<=`, `>=`, `<>`, `||`.
    Punct2,
    Punct,
    /// Whitespace / comment run; kept so raw slices can be reproduced.
    Space,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub kind: Kind,
    pub start: usize,
    pub end: usize,
}

impl Token {
    pub fn text<'a>(&self, src: &'a str) -> &'a str {
        &src[self.start..self.end]
    }
}

/// Tokenise `src`. Never fails: an unknown byte becomes a one-byte `Punct`, so
/// callers see the text rather than an error they cannot act on.
pub fn lex(src: &str) -> Vec<Token> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let start = i;
        let b = bytes[i];
        let kind = if b.is_ascii_whitespace() {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            Kind::Space
        } else if b == b'-' && bytes.get(i + 1) == Some(&b'-') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            Kind::Space
        } else if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            Kind::Space
        } else if b == b'\'' {
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\'' {
                    if bytes.get(i + 1) == Some(&b'\'') {
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                i += 1;
            }
            Kind::String
        } else if b == b'"' || b == b'`' {
            let quote = b;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == quote {
                    if bytes.get(i + 1) == Some(&quote) {
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                i += 1;
            }
            Kind::QuotedIdent
        } else if b.is_ascii_alphabetic() || b == b'_' {
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'$')
            {
                i += 1;
            }
            Kind::Ident
        } else if b.is_ascii_digit() {
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || bytes[i] == b'.'
                    || bytes[i] == b'e'
                    || bytes[i] == b'E'
                    || bytes[i] == b'+'
                    || bytes[i] == b'-')
            {
                // Stop a number at `-` unless it directly follows an exponent.
                if (bytes[i] == b'+' || bytes[i] == b'-')
                    && !matches!(bytes.get(i.wrapping_sub(1)), Some(b'e') | Some(b'E'))
                {
                    break;
                }
                i += 1;
            }
            Kind::Number
        } else if (b == b'<' && bytes.get(i + 1) == Some(&b'-'))
            || (matches!(b, b'-' | b'<' | b'>' | b'!')
                && matches!(bytes.get(i + 1), Some(b'>') | Some(b'=')))
        {
            i += 2;
            Kind::Punct2
        } else if b == b'|' && bytes.get(i + 1) == Some(&b'|') {
            i += 2;
            Kind::Punct2
        } else if b == b':' && bytes.get(i + 1) == Some(&b':') {
            i += 2;
            Kind::Punct2
        } else {
            i += 1;
            Kind::Punct
        };
        out.push(Token {
            kind,
            start,
            end: i,
        });
    }
    out
}

/// Tokens with whitespace and comments removed.
pub fn significant(src: &str) -> Vec<Token> {
    lex(src)
        .into_iter()
        .filter(|t| t.kind != Kind::Space)
        .collect()
}

/// `true` when `tok` is the case-insensitive keyword `word`.
pub fn is_keyword(src: &str, tok: &Token, word: &str) -> bool {
    tok.kind == Kind::Ident && tok.text(src).eq_ignore_ascii_case(word)
}

/// Position of the closer matching the opener at `open` (`(`→`)`, `[`→`]`).
pub fn matching(src: &str, toks: &[Token], open: usize) -> Option<usize> {
    let opener = toks.get(open)?.text(src);
    let closer = match opener {
        "(" => ")",
        "[" => "]",
        _ => return None,
    };
    let mut depth = 0usize;
    for (i, t) in toks.iter().enumerate().skip(open) {
        let text = t.text(src);
        if text == opener {
            depth += 1;
        } else if text == closer {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Position of the `)` matching the `(` at `open`.
pub fn matching_paren(src: &str, toks: &[Token], open: usize) -> Option<usize> {
    debug_assert_eq!(toks.get(open).map(|t| t.text(src)), Some("("));
    matching(src, toks, open)
}

/// Split token indices into top-level comma-separated groups.
///
/// Returns `(start_index, end_index_exclusive)` pairs, trimmed of whitespace.
pub fn split_top_level_commas(src: &str, toks: &[Token]) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, t) in toks.iter().enumerate() {
        match t.text(src) {
            "(" | "[" => depth += 1,
            ")" | "]" => depth = depth.saturating_sub(1),
            "," if depth == 0 => {
                groups.push(trim(src, toks, start, i));
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < toks.len() {
        groups.push(trim(src, toks, start, toks.len()));
    }
    groups
        .into_iter()
        .filter(|(a, b)| b > a)
        .collect()
}

fn trim(src: &str, toks: &[Token], mut a: usize, mut b: usize) -> (usize, usize) {
    while a < b && toks[a].kind == Kind::Space {
        a += 1;
    }
    while b > a && toks[b - 1].kind == Kind::Space {
        b -= 1;
    }
    let _ = src;
    (a, b)
}

/// Unquote an identifier token (`"Foo"` → `Foo`, with `""` → `"`).
pub fn ident_value(src: &str, tok: &Token) -> String {
    let text = tok.text(src);
    if tok.kind == Kind::QuotedIdent && text.len() >= 2 {
        let q = &text[..1];
        let inner = &text[1..text.len() - 1];
        inner.replace(&format!("{q}{q}"), q)
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_and_comments_are_single_tokens() {
        let src = "a 'x''y' -- note\nb /* c */ d";
        let toks = significant(src);
        let texts: Vec<&str> = toks.iter().map(|t| t.text(src)).collect();
        assert_eq!(texts, vec!["a", "'x''y'", "b", "d"]);
    }

    #[test]
    fn arrows_lex_as_pairs() {
        let src = "(a)-[e:T]->(b)<-[f]-(c)";
        let toks = significant(src);
        let texts: Vec<&str> = toks.iter().map(|t| t.text(src)).collect();
        assert!(texts.contains(&"->"));
        assert!(texts.contains(&"<-"));
    }

    #[test]
    fn nested_parens_match() {
        let src = "(a WHERE (x = 1 OR y = 2))";
        let toks = significant(src);
        assert_eq!(matching_paren(src, &toks, 0), Some(toks.len() - 1));
        assert_eq!(matching(src, &toks, 3), Some(toks.len() - 2));
    }

    #[test]
    fn arrows_lex_in_both_directions() {
        let src = "<-";
        let toks = significant(src);
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].text(src), "<-");
    }

    #[test]
    fn quoted_ident_unquotes() {
        let src = "\"Account\"";
        let toks = significant(src);
        assert_eq!(ident_value(src, &toks[0]), "Account");
    }
}
