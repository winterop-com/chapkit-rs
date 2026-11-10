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
    /// Parse a monthly or weekly period in any form chap-core 2.7.0 reads
    /// (`TimePeriod.parse(clean_timestring(s))`):
    ///
    /// - months: `2024-01`, `202401`, `2024-1`;
    /// - Monday weeks: `2024W03`, `2024-W03`, `2024W3`, and a 7-day date range starting on a Monday,
    ///   `2024-01-15/2024-01-21` (chap-core's CSV export);
    /// - Sunday weeks: `2024SunW03`, `2024-S03`, and a 7-day date range starting on a Sunday.
    ///
    /// A week must exist in its ISO year (`2021W53` does not). `None` for anything else, including
    /// days and years, which chap-core reads but chapkit models do not take.
    pub fn parse(s: &str) -> Option<Period> {
        let s = s.trim();
        // Fast path for the common monthly forms, called once per row on large frames.
        if let Some(p) = Self::parse_monthly_digits(s.as_bytes()) {
            return Some(p);
        }
        if let Some((start, end)) = s.split_once('/') {
            return Self::parse_week_range(start.trim(), end.trim());
        }
        let num = |t: &str| t.trim().parse::<u32>().ok();
        let year = |t: &str| t.trim().parse::<i32>().ok();
        for (sep, sunday) in [("-S", true), ("SunW", true), ("-W", false), ("W", false)] {
            if let Some((y, w)) = s.split_once(sep) {
                let (year, week) = (year(y)?, num(w)?);
                return (1..=iso_weeks(year))
                    .contains(&week)
                    .then_some(Period::Week { year, week, sunday });
            }
        }
        // Byte offsets below are only safe on ASCII input.
        if !s.is_ascii() {
            return None;
        }
        let (y, m) = match s.split_once('-') {
            Some((y, m)) if y.len() == 4 && (1..=2).contains(&m.len()) => (y, m),
            None if s.len() == 6 => (s.get(..4)?, s.get(4..)?),
            _ => return None,
        };
        if !m.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let (year, month) = (year(y)?, num(m)?);
        (1..=12)
            .contains(&month)
            .then_some(Period::Month { year, month })
    }

    /// A week written as its first and last day (`YYYY-MM-DD/YYYY-MM-DD`, 7 days, starting on a
    /// Monday or a Sunday). As chap-core numbers it: the ISO year and week of the first day.
    fn parse_week_range(start: &str, end: &str) -> Option<Period> {
        let (a, b) = (days(start)?, days(end)?);
        if b - a != 6 {
            return None;
        }
        // 1970-01-01 was a Thursday: 0 = Monday.
        let weekday = (a + 3).rem_euclid(7);
        let monday = match weekday {
            0 => a,
            6 => a - 6,
            _ => return None,
        };
        // The ISO week's Thursday decides its year.
        let (year, _, _) = civil(monday + 3);
        let week = u32::try_from((monday - week1_monday(year)) / 7 + 1).ok()?;
        Some(Period::Week {
            year,
            week,
            sunday: weekday == 6,
        })
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

    /// The next period: the next month, or the next ISO week (years have 52 or 53). `None` for
    /// Sunday-start weeks, whose year lengths are not implemented.
    pub fn succ(self) -> Option<Period> {
        match self {
            Period::Month { year, month: 12 } => Some(Period::Month {
                year: year + 1,
                month: 1,
            }),
            Period::Month { year, month } => Some(Period::Month {
                year,
                month: month + 1,
            }),
            Period::Week { sunday: true, .. } => None,
            Period::Week { year, week, .. } if week >= iso_weeks(year) => Some(Period::Week {
                year: year + 1,
                week: 1,
                sunday: false,
            }),
            Period::Week { year, week, .. } => Some(Period::Week {
                year,
                week: week + 1,
                sunday: false,
            }),
        }
    }
}

/// Days since 1970-01-01 of a `YYYY-MM-DD` (or `YYYYMMDD`) date; `None` for an invalid date.
fn days(s: &str) -> Option<i64> {
    fn digits(t: &str) -> Option<&str> {
        t.bytes().all(|b| b.is_ascii_digit()).then_some(t)
    }
    let (y, m, d) = match s.len() {
        10 if s.get(4..5) == Some("-") && s.get(7..8) == Some("-") => {
            (s.get(..4)?, s.get(5..7)?, s.get(8..)?)
        }
        8 => (s.get(..4)?, s.get(4..6)?, s.get(6..)?),
        _ => return None,
    };
    let (y, m, d): (i64, i64, i64) = (
        digits(y)?.parse().ok()?,
        digits(m)?.parse().ok()?,
        digits(d)?.parse().ok()?,
    );
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let max = *month_days.get(usize::try_from(m - 1).ok()?)?;
    (1..=max).contains(&d).then(|| days_from_civil(y, m, d))
}

/// Days since 1970-01-01 of a valid date (Howard Hinnant's days_from_civil).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The (year, month, day) of a day count since 1970-01-01 (Hinnant's civil_from_days).
fn civil(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y as i32, m, d)
}

/// Day count of the Monday of ISO week 1 of `year`.
fn week1_monday(year: i32) -> i64 {
    let jan4 = days_from_civil(i64::from(year), 1, 4);
    jan4 - (jan4 + 3).rem_euclid(7)
}

/// Weekday of January 1st of `year`, 0 = Monday (Gauss's algorithm).
fn jan1_weekday(year: i32) -> i32 {
    let y = year - 1;
    (y.rem_euclid(4) * 5 + y.rem_euclid(100) * 4 + y.rem_euclid(400) * 6) % 7
}

/// Number of ISO weeks in `year`: 53 when it starts on a Thursday, or on a Wednesday in a leap year.
pub fn iso_weeks(year: i32) -> u32 {
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    match jan1_weekday(year) {
        3 => 53,
        2 if leap => 53,
        _ => 52,
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
            Period::parse("2020-W53"),
            Some(Period::Week {
                year: 2020,
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

    /// Every form chap-core 2.7.0 reads as a month or week, and some it rejects: each case is
    /// chap-core's own answer (`TimePeriod.parse(clean_timestring(s))`, as its `id`; months as
    /// `YYYY-MM`). The date ranges after the fixed cases are random weeks from 1995 to 2040.
    #[test]
    fn parses_what_chap_core_parses() {
        let cases: &[(&str, Option<&str>)] = &[
            ("2024-01", Some("2024-01")),
            ("202401", Some("2024-01")),
            ("2024-1", Some("2024-01")),
            ("2024-9", Some("2024-09")),
            ("2024-12", Some("2024-12")),
            (" 2024-03 ", Some("2024-03")),
            ("2024W03", Some("2024W03")),
            ("2024W3", Some("2024W03")),
            ("2024-W03", Some("2024W03")),
            ("2024-W3", Some("2024W03")),
            ("2024-S03", Some("2024SunW03")),
            ("2024SunW03", Some("2024SunW03")),
            ("2024SunW3", Some("2024SunW03")),
            ("2019 W01", Some("2019W01")),
            ("2020W53", Some("2020W53")),
            ("2020-W53", Some("2020W53")),
            ("2021W53", None),
            ("2015W53", Some("2015W53")),
            ("2026W53", Some("2026W53")),
            ("2024W54", None),
            ("2024W00", None),
            ("2024-13", None),
            ("2024-00", None),
            ("2024", None),
            ("2024-01-01", None),
            ("2019-01-07/2019-01-13", Some("2019W02")),
            ("2019-01-06/2019-01-12", Some("2019SunW01")),
            ("2020-12-28/2021-01-03", Some("2020W53")),
            ("2021-01-03/2021-01-09", Some("2020SunW53")),
            ("2024-01-01 / 2024-01-07", Some("2024W01")),
            ("2024-01-01/2024-01-31", None),
            ("2024-01-01/2024-01-08", None),
            ("2024-01-03/2024-01-09", None),
            ("2018-12-31/2019-01-06", Some("2019W01")),
            ("2018-12-30/2019-01-05", Some("2018SunW52")),
            ("2016-01-03/2016-01-09", Some("2015SunW53")),
            ("2016-01-04/2016-01-10", Some("2016W01")),
            ("2027-01-03/2027-01-09", Some("2026SunW53")),
            ("2021-01-01/2021-01-07", None),
            ("2024-01-14/2024-01-20", Some("2024SunW02")),
            ("2030-06-02/2030-06-08", Some("2030SunW22")),
            ("2001-07-01/2001-07-07", Some("2001SunW26")),
            ("2003-06-09/2003-06-15", Some("2003W24")),
            ("2000-03-12/2000-03-18", Some("2000SunW10")),
            ("2014-03-31/2014-04-06", Some("2014W14")),
            ("2033-11-21/2033-11-27", Some("2033W47")),
            ("2016-08-01/2016-08-07", Some("2016W31")),
            ("2033-01-31/2033-02-06", Some("2033W05")),
            ("2006-02-05/2006-02-11", Some("2006SunW05")),
            ("2000-07-16/2000-07-22", Some("2000SunW28")),
            ("2030-07-29/2030-08-04", Some("2030W31")),
            ("2014-10-27/2014-11-02", Some("2014W44")),
            ("2006-12-11/2006-12-17", Some("2006W50")),
            ("2007-12-09/2007-12-15", Some("2007SunW49")),
            ("2022-09-04/2022-09-10", Some("2022SunW35")),
            ("2011-03-14/2011-03-20", Some("2011W11")),
            ("2011-11-07/2011-11-13", Some("2011W45")),
            ("2000-08-13/2000-08-19", Some("2000SunW32")),
            ("2013-06-17/2013-06-23", Some("2013W25")),
            ("2033-05-08/2033-05-14", Some("2033SunW18")),
            ("2036-10-05/2036-10-11", Some("2036SunW40")),
            ("2035-08-27/2035-09-02", Some("2035W35")),
            ("2017-04-09/2017-04-15", Some("2017SunW14")),
            ("2016-11-21/2016-11-27", Some("2016W47")),
            ("2021-12-05/2021-12-11", Some("2021SunW48")),
            ("2025-10-19/2025-10-25", Some("2025SunW42")),
            ("2020-10-25/2020-10-31", Some("2020SunW43")),
            ("2001-07-23/2001-07-29", Some("2001W30")),
            ("2032-07-05/2032-07-11", Some("2032W28")),
            ("2025-09-08/2025-09-14", Some("2025W37")),
            ("2038-11-08/2038-11-14", Some("2038W45")),
            ("2001-12-16/2001-12-22", Some("2001SunW50")),
            ("2023-02-20/2023-02-26", Some("2023W08")),
            ("2026-05-31/2026-06-06", Some("2026SunW22")),
            ("2035-12-03/2035-12-09", Some("2035W49")),
            ("2003-05-25/2003-05-31", Some("2003SunW21")),
            ("2037-07-12/2037-07-18", Some("2037SunW28")),
            ("2000-10-30/2000-11-05", Some("2000W44")),
            ("2022-10-09/2022-10-15", Some("2022SunW40")),
        ];
        for (input, want) in cases {
            let got = Period::parse(input).map(|p| p.to_string());
            assert_eq!(got.as_deref(), *want, "{input}");
        }
    }

    #[test]
    fn period_of_year_pools_week_53() {
        assert_eq!(Period::parse("2020W53").unwrap().period_of_year(), 52);
        assert_eq!(Period::parse("2020-07").unwrap().period_of_year(), 7);
    }

    #[test]
    fn succ_crosses_years_and_week_53() {
        let next = |s: &str| Period::parse(s).unwrap().succ().unwrap().to_string();
        assert_eq!(next("2020-12"), "2021-01");
        assert_eq!(next("2020-03"), "2020-04");
        assert_eq!(next("2020W52"), "2020W53");
        assert_eq!(next("2020W53"), "2021W01");
        assert_eq!(next("2021W52"), "2022W01");
        assert_eq!(next("2015W52"), "2015W53");
        assert_eq!(Period::parse("2020SunW03").unwrap().succ(), None);
        let long: Vec<i32> = (2000..2031).filter(|y| iso_weeks(*y) == 53).collect();
        assert_eq!(long, [2004, 2009, 2015, 2020, 2026]);
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
