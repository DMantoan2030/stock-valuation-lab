"""Checks the 10-K reading rules on small made-up filings."""
from valuation import filing

LONG = "This paragraph is long enough to count as real prose in the filing. " * 4

FAKE_10K = f"""
Table of Contents
Item 1. Business | 3
Item 1A. Risk Factors | 9
Item 7. Management's Discussion and Analysis | 20
Item 1. Business
In this report, the terms "we" and "our" mean Example Corp and its subsidiaries, as used throughout. {LONG}
Example Corp makes widgets for hospitals and sells maintenance services worldwide. {LONG}
Our strategy is to grow services faster than hardware by bundling maintenance contracts. {LONG}
Item 1A. Risk Factors
Demand for our widgets depends on hospital budgets.
{LONG}
Item 7. Management's Discussion and Analysis
Overview
This discussion should be read in conjunction with the financial statements. {LONG}
Revenue grew because of higher service volumes in Europe. {LONG}
Challenges
Rising component costs and tariffs put pressure on margins this year. {LONG}
| | 2026 | | 2025
| Widgets | $ | 600 | | $ | 550
| Services | 400 | | 350
| Total revenues | $ | 1,000 | | $ | 900
Item 8. Financial Statements
"""


def test_section_skips_table_of_contents():
    business = filing.extract_section(FAKE_10K, "1")
    assert "makes widgets for hospitals" in business
    assert "Risk Factors" not in business.split("\n", 1)[1]   # stops at Item 1A


def test_boilerplate_is_skipped_but_description_kept():
    paras = filing.paragraphs(filing.extract_section(FAKE_10K, "1"))
    assert paras[0].startswith("Example Corp makes widgets")


def test_struggles_come_from_challenges_heading():
    heading, paras = filing.paragraphs_after_heading(filing.extract_section(FAKE_10K, "7"),
                                                     filing.STRUGGLE_HEADINGS, 3)
    assert heading == "Challenges"
    assert paras[0].startswith("Rising component costs")


def test_revenue_table_rows_must_add_up():
    rows = filing.revenue_table(filing.extract_section(FAKE_10K, "7"))
    assert rows[0] == ["", "2026", "2025"]
    assert rows[-1][0] == "Total revenues"
    assert filing.table_total(rows) == 1000


def test_revenue_table_rejected_when_rows_do_not_add_up():
    bad = FAKE_10K.replace("| Services | 400 |", "| Services | 999 |")
    assert filing.revenue_table(filing.extract_section(bad, "7")) == []


def test_risk_headlines_come_from_bold_text():
    html = ("<p><b>Demand for our widgets depends on hospital budgets.</b></p>"
            "<p>Hospitals may delay purchases when budgets tighten, which could reduce sales.</p>")
    text, emphasized = filing.html_to_text_and_emphasis(html)
    assert filing.risk_headlines(text, emphasized) == ["Demand for our widgets depends on hospital budgets."]
