//! Chap time period ids: monthly (`2024-01`, `202401`) and weekly
//! (`2024W03`, `2024-W03`, `2024-S03`, `2024SunW03`).

use std::fmt;

/// A parsed monthly or weekly period.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Period {
    /// Calendar month (`month` is 1-12).
    Month { year: i32, month: u32 },
    /// Week of year (`week` is 1-53). `sunday` marks Sunday-start weeks.
    Week { year: i32, week: u32, sunday: bool },
}

impl Period {
    /// Parse a chap period id. Returns `None` for anything else (years, days, ranges).
    pub fn parse(s: &str) -> Option<Period> {
        let s = s.trim();
        // Fast path for the common monthly forms, called once per row on large frames.
        if let Some(p) = Self::parse_monthly_digits(s.as_bytes()) {
            return Some(p);
        }
        let num = |t: &str| t.parse::<u32>().ok();
        let year = |t: &str| t.parse::<i32>().ok();
        for (sep, sunday) in [("-S", true), ("SunW", true), ("-W", false), ("W", false)] {
            if let Some((y, w)) = s.split_once(sep) {
                let (year, week) = (year(y)?, num(w)?);
                return (1..=53)
                    .contains(&week)
                    .then_some(Period::Week { year, week, sunday });
            }
        }
        // Byte offsets below are only safe on ASCII input.
        if !s.is_ascii() {
            return None;
        }
        let (y, m) = match s.split_once('-') {
            Some((y, m)) if y.len() == 4 && m.len() == 2 => (y, m),
            None if s.len() == 6 => (s.get(..4)?, s.get(4..)?),
            _ => return None,
        };
        let (year, month) = (year(y)?, num(m)?);
        (1..=12)
            .contains(&month)
            .then_some(Period::Month { year, month })
    }

    /// `YYYY-MM` or `YYYYMM` made of ASCII digits; `None` for anything else.
    fn parse_monthly_digits(b: &[u8]) -> Option<Period> {
        let digit = |c: u8| c.is_ascii_digit().then(|| u32::from(c - b'0'));
        let (y, m) = match b {
            [y0, y1, y2, y3, b'-', m0, m1] | [y0, y1, y2, y3, m0, m1] => {
                ([*y0, *y1, *y2, *y3], [*m0, *m1])
            }
            _ => return None,
        };
        let mut year = 0u32;
        for c in y {
            year = year * 10 + digit(c)?;
        }
        let month = digit(m[0])? * 10 + digit(m[1])?;
        (1..=12).contains(&month).then_some(Period::Month {
            year: i32::try_from(year).ok()?,
            month,
        })
    }

    /// Number of distinct periods of year: 12 for months, 52 for weeks.
    pub fn periods_per_year(self) -> u32 {
        match self {
            Period::Month { .. } => 12,
            Period::Week { .. } => 52,
        }
    }

    /// Month (1-12) or week (1-52) of the year. Week 53 counts as week 52, as in chap-core.
    pub fn period_of_year(self) -> u32 {
        match self {
            Period::Month { month, .. } => month,
            Period::Week { week, .. } => week.min(52),
        }
    }
}

impl fmt::Display for Period {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Period::Month { year, month } => write!(f, "{year}-{month:02}"),
            Period::Week {
                year,
                week,
                sunday: false,
            } => write!(f, "{year}W{week:02}"),
            Period::Week {
                year,
                week,
                sunday: true,
            } => write!(f, "{year}SunW{week:02}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chap_ids() {
        assert_eq!(
            Period::parse("2024-01"),
            Some(Period::Month {
                year: 2024,
                month: 1
            })
        );
        assert_eq!(
            Period::parse("202412"),
            Some(Period::Month {
                year: 2024,
                month: 12
            })
        );
        assert_eq!(
            Period::parse("2024W03"),
            Some(Period::Week {
                year: 2024,
                week: 3,
                sunday: false
            })
        );
        assert_eq!(
            Period::parse("2024-W53"),
            Some(Period::Week {
                year: 2024,
                week: 53,
                sunday: false
            })
        );
        assert_eq!(
            Period::parse("2024-S07"),
            Some(Period::Week {
                year: 2024,
                week: 7,
                sunday: true
            })
        );
        assert_eq!(
            Period::parse("2024SunW07"),
            Some(Period::Week {
                year: 2024,
                week: 7,
                sunday: true
            })
        );
        assert_eq!(Period::parse("2024-13"), None);
        assert_eq!(Period::parse("2024"), None);
        assert_eq!(Period::parse("20240101"), None);
        // Non-ASCII input whose byte length matches a period id must not panic.
        assert_eq!(Period::parse("123é4"), None);
        // The monthly fast path agrees with the general rules.
        assert_eq!(
            Period::parse(" 2024-03 "),
            Some(Period::Month {
                year: 2024,
                month: 3
            })
        );
        assert_eq!(
            Period::parse("1850-12"),
            Some(Period::Month {
                year: 1850,
                month: 12
            })
        );
        assert_eq!(Period::parse("2024-00"), None);
        assert_eq!(Period::parse("2024-1x"), None);
        assert_eq!(Period::parse("202400"), None);
        assert_eq!(Period::parse("2024+01"), None);
        assert_eq!(Period::parse("2024é"), None);
        assert_eq!(Period::parse("é2024-01"), None);
        assert_eq!(Period::parse("2024Wé"), None);
    }

    #[test]
    fn period_of_year_pools_week_53() {
        assert_eq!(Period::parse("2020W53").unwrap().period_of_year(), 52);
        assert_eq!(Period::parse("2020-07").unwrap().period_of_year(), 7);
    }

    #[test]
    fn orders_chronologically() {
        let mut v: Vec<Period> = ["2024-02", "2023-12", "2024-01"]
            .iter()
            .map(|s| Period::parse(s).unwrap())
            .collect();
        v.sort();
        let s: Vec<String> = v.iter().map(ToString::to_string).collect();
        assert_eq!(s, ["2023-12", "2024-01", "2024-02"]);
    }
}
