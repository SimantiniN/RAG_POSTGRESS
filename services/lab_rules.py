
import pandas as pd


def screen_labs(labs: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent preliminary rules to individual lab results."""
    columns = [
        "charttime", "itemid", "lab_name", "valuenum", "unit",
        "ref_range_lower", "ref_range_upper", "source_flag"
    ]

    if labs.empty:
        return pd.DataFrame(
            columns=columns + ["screening_status", "screening_reason"]
        )

    results = labs.copy()

    for col in [
        "valuenum", "ref_range_lower", "ref_range_upper"
    ]:
        results[col] = pd.to_numeric(results[col], errors="coerce")

    def assess(row):
        reasons = []
        value = row.get("valuenum")
        low = row.get("ref_range_lower")
        high = row.get("ref_range_upper")
        flag = row.get("source_flag")
        unit = row.get("unit")
        timestamp = row.get("charttime")

        if pd.isna(value):
            reasons.append("Numeric result unavailable")
        else:
            if pd.notna(low) and value < low:
                reasons.append(f"Below recorded reference range ({low})")
            if pd.notna(high) and value > high:
                reasons.append(f"Above recorded reference range ({high})")

        if pd.notna(flag) and str(flag).strip():
            reasons.append(f"Source flag: {flag}")

        if pd.isna(low) and pd.isna(high):
            reasons.append("Reference range unavailable")

        if pd.isna(unit) or not str(unit).strip():
            reasons.append("Unit missing")

        if pd.isna(timestamp):
            reasons.append("Timestamp missing")

        if any(
            reason.startswith(("Below recorded", "Above recorded", "Source flag:"))
            for reason in reasons
        ):
            status = "REVIEW"
        elif reasons:
            status = "DATA_QUALITY_REVIEW"
        else:
            status = "NO_FLAG_FROM_AVAILABLE_RULES"

        return pd.Series({
            "screening_status": status,
            "screening_reason": "; ".join(reasons) if reasons
            else "No flag from the available rules"
        })

    assessed = results.apply(assess, axis=1)
    return pd.concat([results, assessed], axis=1)


def screen_lab_trends(labs: pd.DataFrame) -> pd.DataFrame:
    """Identify changes between first and latest numeric results."""
    if labs.empty:
        return pd.DataFrame()

    data = labs.copy()
    data["valuenum"] = pd.to_numeric(data["valuenum"], errors="coerce")
    data = data.dropna(subset=["valuenum", "charttime"])

    if data.empty:
        return pd.DataFrame()

    data = data.sort_values("charttime")

    summary = data.groupby(
        ["itemid", "lab_name"], dropna=False
    ).agg(
        first_time=("charttime", "first"),
        latest_time=("charttime", "last"),
        first_value=("valuenum", "first"),
        latest_value=("valuenum", "last"),
        result_count=("valuenum", "count"),
    ).reset_index()

    summary["absolute_change"] = (
        summary["latest_value"] - summary["first_value"]
    )

    summary["trend_note"] = summary.apply(
        lambda r: (
            "Multiple recorded values; review trend in clinical context"
            if r["result_count"] > 1
            else "Only one numeric result; trend cannot be assessed"
        ),
        axis=1,
    )

    return summary
