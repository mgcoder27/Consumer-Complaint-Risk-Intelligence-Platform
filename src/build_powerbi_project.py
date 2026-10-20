"""Generate a native PBIP report and semantic model from Stage 7 CSV exports."""

import argparse
import json
from pathlib import Path
import re
import uuid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORTS = PROJECT_ROOT / "data" / "processed" / "powerbi"
DEFAULT_OUTPUT = PROJECT_ROOT / "powerbi"
REPORT_NAME = "ConsumerComplaintDashboard.Report"
MODEL_NAME = "ConsumerComplaintDashboard.SemanticModel"
PROJECT_NAME = "ConsumerComplaintDashboard"
NAMESPACE = uuid.UUID("c76c8412-0874-4653-bb19-6c4f6b849afa")

TABLES = {
    "Monthly Volume": {
        "file": "monthly_volume_trend.csv",
        "columns": {
            "month": "string",
            "complaint_count": "int64",
            "closed_outcome_count": "int64",
            "monetary_relief_count": "int64",
            "monetary_relief_rate_pct": "double",
            "previous_month_count": "double",
            "month_over_month_change": "double",
            "month_over_month_change_pct": "double",
        },
        "measures": {
            "Total Complaints": (
                "SUM('Monthly Volume'[complaint_count])",
                "#,##0",
            ),
            "Monetary Relief Count": (
                "SUM('Monthly Volume'[monetary_relief_count])",
                "#,##0",
            ),
            "Closed Outcomes": (
                "SUM('Monthly Volume'[closed_outcome_count])",
                "#,##0",
            ),
            "Monetary Relief Rate": (
                "DIVIDE([Monetary Relief Count], [Closed Outcomes])",
                "0.00%",
            ),
        },
    },
    "Product Summary": {
        "file": "product_summary.csv",
        "columns": {
            "product": "string",
            "complaint_count": "int64",
            "closed_outcome_count": "int64",
            "monetary_relief_count": "int64",
            "monetary_relief_rate_pct": "double",
            "complaint_volume_rank": "int64",
        },
        "measures": {
            "Product Complaints": (
                "SUM('Product Summary'[complaint_count])",
                "#,##0",
            ),
        },
    },
    "Issue Summary": {
        "file": "issue_summary.csv",
        "columns": {
            "issue": "string",
            "complaint_count": "int64",
            "closed_outcome_count": "int64",
            "monetary_relief_count": "int64",
            "monetary_relief_rate_pct": "double",
            "complaint_volume_rank": "int64",
        },
        "measures": {
            "Issue Complaints": (
                "SUM('Issue Summary'[complaint_count])",
                "#,##0",
            ),
        },
    },
    "Risk Categories": {
        "file": "risk_category_summary.csv",
        "columns": {
            "risk_category": "string",
            "complaint_count": "int64",
            "closed_outcome_count": "int64",
            "monetary_relief_count": "int64",
            "monetary_relief_rate_pct": "double",
            "complaint_volume_rank": "int64",
        },
        "measures": {
            "Category Complaints": (
                "SUM('Risk Categories'[complaint_count])",
                "#,##0",
            ),
            "Category Relief Count": (
                "SUM('Risk Categories'[monetary_relief_count])",
                "#,##0",
            ),
        },
    },
    "Company Relief Rates": {
        "file": "company_relief_rates.csv",
        "columns": {
            "company": "string",
            "complaint_count": "int64",
            "closed_outcome_count": "int64",
            "monetary_relief_count": "int64",
            "monetary_relief_rate_pct": "double",
            "complaint_volume_rank": "int64",
            "relief_rate_rank_min_100_outcomes": "double",
        },
        "measures": {},
    },
    "Company Focus": {
        "file": "company_relief_rates.csv",
        "columns": {
            "company": "string",
            "complaint_count": "int64",
            "closed_outcome_count": "int64",
            "monetary_relief_count": "int64",
            "monetary_relief_rate_pct": "double",
            "complaint_volume_rank": "int64",
            "relief_rate_rank_min_100_outcomes": "double",
        },
        "measures": {},
        "top_companies_only": True,
    },
    "State Complaints": {
        "file": "state_complaints_per_capita.csv",
        "columns": {
            "state": "string",
            "complaint_count": "int64",
            "closed_outcome_count": "int64",
            "monetary_relief_count": "int64",
            "monetary_relief_rate_pct": "double",
            "population": "int64",
            "complaints_per_100k": "double",
            "complaint_rate_rank": "double",
            "relief_rate_rank_min_100_outcomes": "double",
        },
        "measures": {
            "State Complaints": (
                "SUM('State Complaints'[complaint_count])",
                "#,##0",
            ),
        },
    },
    "Business Recommendations": {
        "file": "business_recommendations.csv",
        "columns": {
            "priority": "int64",
            "recommendation": "string",
            "evidence": "string",
            "action": "string",
        },
        "measures": {},
    },
    "Model Drivers": {
        "file": "model_drivers.csv",
        "columns": {
            "driver_rank": "int64",
            "model": "string",
            "feature": "string",
            "coefficient": "double",
            "direction": "string",
            "status": "string",
        },
        "measures": {},
    },
}


def stable_id(value: str) -> str:
    """Return a reproducible Power BI lineage or object identifier."""
    return str(uuid.uuid5(NAMESPACE, value))


def tmdl_identifier(value: str) -> str:
    """Quote TMDL names and escape embedded apostrophes."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return value
    return "'" + value.replace("'", "''") + "'"


def write_json(path: Path, contents: dict) -> None:
    """Write stable, UTF-8 JSON files for PBIP/PBIR definitions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contents, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_exports(export_dir: Path) -> None:
    """Fail early if any Stage 7 export required by the report is missing."""
    required = {table["file"] for table in TABLES.values()}
    required.update(
        {
            "business_recommendations.csv",
            "model_drivers.csv",
        }
    )
    missing = sorted(filename for filename in required if not (export_dir / filename).is_file())
    if missing:
        raise FileNotFoundError(
            "Generate Stage 7 exports before building the report. Missing: "
            + ", ".join(missing)
        )


def make_model_definition(output_dir: Path, data_folder: Path) -> None:
    """Write a self-contained TMDL model with a local data-folder parameter."""
    model_dir = output_dir / MODEL_NAME
    definition_dir = model_dir / "definition"
    tables_dir = definition_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        model_dir / "definition.pbism",
        {"version": "4.1", "settings": {"qnaEnabled": False}},
    )
    (definition_dir / "database.tmdl").write_text(
        "database\n\tcompatibilityLevel: 1601\n",
        encoding="utf-8",
    )
    model_lines = [
        "model Model",
        "\tculture: en-US",
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
        "\tdiscourageImplicitMeasures",
    ]
    for table_name in TABLES:
        model_lines.append(f"\tref table {tmdl_identifier(table_name)}")
    model_lines.append("")
    (definition_dir / "model.tmdl").write_text(
        "\n".join(model_lines),
        encoding="utf-8",
    )

    escaped_folder = str(data_folder).replace('"', '""')
    (definition_dir / "expressions.tmdl").write_text(
        "\n".join(
            [
                f'expression DataFolder = "{escaped_folder}" meta '
                '[IsParameterQuery = true, Type = "Text", '
                "IsParameterQueryRequired = true]",
                f"\tlineageTag: {stable_id('expression:DataFolder')}",
                "\tannotation PBI_ResultType = Text",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for table_name, specification in TABLES.items():
        write_table(
            tables_dir / f"{table_name}.tmdl",
            table_name,
            specification,
        )


def write_table(path: Path, table_name: str, specification: dict) -> None:
    """Create a TMDL table whose import partition reads one Stage 7 CSV."""
    table_ref = tmdl_identifier(table_name)
    table_lines = [
        f"table {table_ref}",
        f"\tlineageTag: {stable_id(f'table:{table_name}')}",
        "",
    ]
    for measure_name, (formula, format_string) in specification["measures"].items():
        table_lines.extend(
            [
                f"\tmeasure {tmdl_identifier(measure_name)} = {formula}",
                f"\t\tformatString: {format_string}",
                f"\t\tlineageTag: {stable_id(f'measure:{table_name}:{measure_name}')}",
                "",
            ]
        )

    for column_name, data_type in specification["columns"].items():
        aggregation = "none" if data_type == "string" else "sum"
        if column_name.endswith("_rate_pct") or column_name.endswith("_per_100k"):
            aggregation = "none"
        table_lines.extend(
            [
                f"\tcolumn {tmdl_identifier(column_name)}",
                f"\t\tdataType: {data_type}",
                f"\t\tlineageTag: {stable_id(f'column:{table_name}:{column_name}')}",
                f"\t\tsummarizeBy: {aggregation}",
                f"\t\tsourceColumn: {column_name}",
            ]
        )
        if column_name.endswith("_rate_pct") or column_name.endswith("_per_100k"):
            table_lines.append("\t\tformatString: 0.00")
        if column_name in {"month", "state", "product", "issue", "company", "risk_category"}:
            table_lines.extend(
                [
                    "",
                    "\t\tannotation SummarizationSetBy = User",
                ]
            )
        table_lines.append("")

    partition_id = stable_id(f"partition:{table_name}")
    table_lines.extend(
        [
            f"\tpartition {tmdl_identifier(table_name + '-' + partition_id)} = m",
            "\t\tmode: import",
            "\t\tsource =",
            *make_power_query(specification),
            "",
            "\tannotation PBI_ResultType = Table",
            "",
        ]
    )
    path.write_text("\n".join(table_lines), encoding="utf-8")


def make_power_query(specification: dict) -> list[str]:
    """Return an M partition that loads and types one aggregate CSV."""
    filename = specification["file"]
    lines = [
        "\t\t\tlet",
        "\t\t\t\tSource = Csv.Document(",
        f'\t\t\t\t\tFile.Contents(DataFolder & "\\{filename}"),',
        "\t\t\t\t\t[Delimiter = \",\", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]",
        "\t\t\t\t),",
        "\t\t\t\tPromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),",
    ]
    type_specs = []
    for column, data_type in specification["columns"].items():
        power_query_type = {
            "string": "type text",
            "int64": "Int64.Type",
            "double": "type number",
        }[data_type]
        type_specs.append(f'{{"{column}", {power_query_type}}}')
    lines.append(
        "\t\t\t\tTypedColumns = Table.TransformColumnTypes("
    )
    lines.append(
        "\t\t\t\t\tPromotedHeaders, {" + ", ".join(type_specs) + "}"
    )
    lines.append("\t\t\t\t)")
    last_step = "TypedColumns"
    if specification.get("top_companies_only"):
        lines.extend(
            [
                ",",
                "\t\t\t\tEligibleCompanies = Table.SelectRows(TypedColumns, each [relief_rate_rank_min_100_outcomes] <> null),",
                '\t\t\t\tSortedCompanies = Table.Sort(EligibleCompanies, {{"relief_rate_rank_min_100_outcomes", Order.Ascending}}),',
                "\t\t\t\tTopCompanies = Table.FirstN(SortedCompanies, 15)",
            ]
        )
        last_step = "TopCompanies"
    lines.extend(["\t\t\tin", f"\t\t\t\t{last_step}"])
    return lines


def field_reference(table_name: str, column_name: str, measure: bool = False) -> dict:
    """Build a PBIR semantic field reference for a model column or measure."""
    field_type = "Measure" if measure else "Column"
    return {
        field_type: {
            "Expression": {"SourceRef": {"Entity": table_name}},
            "Property": column_name,
        }
    }


def make_projection(table_name: str, column_name: str, measure: bool = False) -> dict:
    """Create a visual projection that binds to a semantic-model field."""
    return {
        "field": field_reference(table_name, column_name, measure=measure),
        "queryRef": f"{table_name}.{column_name}",
        "nativeQueryRef": column_name,
    }


def add_visual(
    page_visuals: Path,
    page_id: str,
    title: str,
    visual_type: str,
    role_projections: dict[str, list[dict]],
    position: tuple[int, int, int, int],
    sort_by: tuple[str, str, bool] | None = None,
    sort_ascending: bool = False,
) -> None:
    """Write one PBIR visual with a data query and an explicit title."""
    visual_name = stable_id(f"visual:{page_id}:{title}")[:20]
    visual_dir = page_visuals / visual_name
    x, y, width, height = position
    query_state = {
        role: {"projections": projections}
        for role, projections in role_projections.items()
    }
    visual = {
        "visualType": visual_type,
        "query": {"queryState": query_state},
        "visualContainerObjects": {
            "title": [
                {
                    "properties": {
                        "show": {
                            "expr": {"Literal": {"Value": "true"}}
                        },
                        "text": {
                            "expr": {
                                "Literal": {
                                    "Value": "'"
                                    + title.replace("'", "''")
                                    + "'"
                                }
                            }
                        },
                    }
                }
            ]
        },
        "drillFilterOtherVisuals": True,
    }
    if sort_by:
        table_name, field_name, is_measure = sort_by
        visual["query"]["sortDefinition"] = {
            "sort": [
                {
                    "field": field_reference(
                        table_name,
                        field_name,
                        measure=is_measure,
                    ),
                    "direction": "Ascending" if sort_ascending else "Descending",
                }
            ],
            "isDefaultSort": True,
        }
    write_json(
        visual_dir / "visual.json",
        {
            "$schema": (
                "https://developer.microsoft.com/json-schemas/fabric/item/report/"
                "definition/visualContainer/2.0.0/schema.json"
            ),
            "name": visual_name,
            "position": {
                "x": x,
                "y": y,
                "z": len(list(page_visuals.glob("*/visual.json"))) * 1000,
                "height": height,
                "width": width,
                "tabOrder": len(list(page_visuals.glob("*/visual.json"))) * 1000,
            },
            "visual": visual,
        },
    )


def card(
    visuals: Path,
    page_id: str,
    title: str,
    table: str,
    measure: str,
    x: int,
    y: int = 20,
    width: int = 285,
) -> None:
    """Add a single-measure KPI card."""
    add_visual(
        visuals,
        page_id,
        title,
        "card",
        {"Values": [make_projection(table, measure, measure=True)]},
        (x, y, width, 115),
    )


def chart(
    visuals: Path,
    page_id: str,
    title: str,
    visual_type: str,
    table: str,
    category: str,
    value_table: str,
    value_measure: str,
    position: tuple[int, int, int, int],
    category_role: str = "Category",
    sort_ascending: bool = False,
    sort_by: tuple[str, str, bool] | None = None,
) -> None:
    """Add a category/value chart with the requested visual type."""
    add_visual(
        visuals,
        page_id,
        title,
        visual_type,
        {
            category_role: [make_projection(table, category)],
            "Y": [make_projection(value_table, value_measure, measure=True)],
        },
        position,
        sort_by=sort_by or (value_table, value_measure, True),
        sort_ascending=sort_ascending,
    )


def table_visual(
    visuals: Path,
    page_id: str,
    title: str,
    table: str,
    columns: list[str],
    position: tuple[int, int, int, int],
    sort_by: tuple[str, str, bool] | None = None,
    sort_ascending: bool = False,
) -> None:
    """Add a table visual with explicit source columns."""
    add_visual(
        visuals,
        page_id,
        title,
        "tableEx",
        {
            "Values": [
                make_projection(table, column)
                for column in columns
            ]
        },
        position,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
    )


def make_page(report_definition: Path, name: str, display_name: str) -> Path:
    """Create the page folder and set up the page canvas and visuals folder."""
    page_dir = report_definition / "pages" / name
    write_json(
        page_dir / "page.json",
        {
            "$schema": (
                "https://developer.microsoft.com/json-schemas/fabric/item/report/"
                "definition/page/1.4.0/schema.json"
            ),
            "name": name,
            "displayName": display_name,
            "displayOption": "FitToPage",
            "height": 720,
            "width": 1280,
        },
    )
    visuals = page_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    return visuals


def make_report_definition(output_dir: Path) -> None:
    """Create a four-page PBIR report with standard built-in visuals."""
    report_dir = output_dir / REPORT_NAME
    definition = report_dir / "definition"
    write_json(
        report_dir / "definition.pbir",
        {
            "$schema": (
                "https://developer.microsoft.com/json-schemas/fabric/item/report/"
                "definitionProperties/2.0.0/schema.json"
            ),
            "version": "4.0",
            "datasetReference": {"byPath": {"path": f"../{MODEL_NAME}"}},
        },
    )
    write_json(
        definition / "report.json",
        {
            "$schema": (
                "https://developer.microsoft.com/json-schemas/fabric/item/report/"
                "definition/report/3.0.0/schema.json"
            ),
            "themeCollection": {
                "baseTheme": {
                    "name": "CY24SU10",
                    "reportVersionAtImport": {
                        "visual": "1.8.97",
                        "report": "2.0.97",
                        "page": "1.3.97",
                    },
                    "type": "SharedResources",
                }
            },
            "settings": {
                "useStylableVisualContainerHeader": True,
                "defaultFilterActionIsDataFilter": True,
            },
        },
    )
    write_json(
        definition / "version.json",
        {
            "$schema": (
                "https://developer.microsoft.com/json-schemas/fabric/item/report/"
                "definition/versionMetadata/1.0.0/schema.json"
            ),
            "version": "2.0.0",
        },
    )

    page_specs = [
        ("OverviewPage", "Overview"),
        ("RiskPage", "Risk Categories"),
        ("CompaniesStatesPage", "Companies & States"),
        ("ModelDriversPage", "Model Drivers & Actions"),
    ]
    write_json(
        definition / "pages" / "pages.json",
        {
            "$schema": (
                "https://developer.microsoft.com/json-schemas/fabric/item/report/"
                "definition/pagesMetadata/1.0.0/schema.json"
            ),
            "pageOrder": [name for name, _ in page_specs],
            "activePageName": "OverviewPage",
        },
    )

    overview = make_page(definition, *page_specs[0])
    card(overview, "OverviewPage", "Total Complaints", "Monthly Volume", "Total Complaints", 20)
    card(overview, "OverviewPage", "Monetary Relief Outcomes", "Monthly Volume", "Monetary Relief Count", 325)
    card(overview, "OverviewPage", "Relief Rate (Closed Outcomes)", "Monthly Volume", "Monetary Relief Rate", 630)
    chart(
        overview,
        "OverviewPage",
        "Monthly Complaint Volume",
        "lineChart",
        "Monthly Volume",
        "month",
        "Monthly Volume",
        "Total Complaints",
        (20, 155, 790, 300),
        sort_ascending=True,
        sort_by=("Monthly Volume", "month", False),
    )
    chart(
        overview,
        "OverviewPage",
        "Complaints by Product",
        "clusteredBarChart",
        "Product Summary",
        "product",
        "Product Summary",
        "Product Complaints",
        (830, 155, 430, 480),
    )
    table_visual(
        overview,
        "OverviewPage",
        "Monthly changes",
        "Monthly Volume",
        ["month", "complaint_count", "month_over_month_change", "month_over_month_change_pct"],
        (20, 475, 790, 160),
    )

    risk = make_page(definition, *page_specs[1])
    chart(
        risk,
        "RiskPage",
        "Complaints by Heuristic Risk Category",
        "clusteredBarChart",
        "Risk Categories",
        "risk_category",
        "Risk Categories",
        "Category Complaints",
        (20, 20, 560, 300),
    )
    table_visual(
        risk,
        "RiskPage",
        "Category volumes and observed relief rates",
        "Risk Categories",
        [
            "risk_category",
            "complaint_count",
            "closed_outcome_count",
            "monetary_relief_count",
            "monetary_relief_rate_pct",
        ],
        (600, 20, 660, 300),
    )
    chart(
        risk,
        "RiskPage",
        "Most Frequent Issues",
        "clusteredBarChart",
        "Issue Summary",
        "issue",
        "Issue Summary",
        "Issue Complaints",
        (20, 350, 1240, 285),
    )

    companies = make_page(definition, *page_specs[2])
    table_visual(
        companies,
        "CompaniesStatesPage",
        "Companies: highest relief rates (minimum 100 closed outcomes)",
        "Company Focus",
        [
            "relief_rate_rank_min_100_outcomes",
            "company",
            "closed_outcome_count",
            "monetary_relief_count",
            "monetary_relief_rate_pct",
        ],
        (20, 20, 660, 570),
        sort_by=("Company Focus", "relief_rate_rank_min_100_outcomes", False),
        sort_ascending=True,
    )
    chart(
        companies,
        "CompaniesStatesPage",
        "Complaint Volume by State",
        "clusteredBarChart",
        "State Complaints",
        "state",
        "State Complaints",
        "State Complaints",
        (710, 20, 550, 350),
    )
    table_visual(
        companies,
        "CompaniesStatesPage",
        "State counts and per-capita fields",
        "State Complaints",
        ["state", "complaint_count", "population", "complaints_per_100k", "monetary_relief_rate_pct"],
        (710, 395, 550, 195),
    )

    drivers = make_page(definition, *page_specs[3])
    table_visual(
        drivers,
        "ModelDriversPage",
        "Model driver status (no fitted model is available)",
        "Model Drivers",
        ["model", "feature", "coefficient", "direction", "status"],
        (20, 20, 1240, 145),
    )
    table_visual(
        drivers,
        "ModelDriversPage",
        "Number-backed compliance review recommendations",
        "Business Recommendations",
        ["priority", "recommendation", "evidence", "action"],
        (20, 190, 1240, 430),
    )


def build_powerbi_project(
    export_dir: Path = DEFAULT_EXPORTS,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Path:
    """Build and return the PBIP file path."""
    validate_exports(export_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    make_model_definition(output_dir, export_dir.resolve())
    make_report_definition(output_dir)
    project_file = output_dir / f"{PROJECT_NAME}.pbip"
    write_json(
        project_file,
        {
            "version": "1.0",
            "artifacts": [
                {"report": {"path": REPORT_NAME}},
                {"semanticModel": {"path": MODEL_NAME}},
            ],
            "settings": {"enableAutoRecovery": True},
        },
    )
    return project_file


def main() -> None:
    """Build the PBIP report from existing Stage 7 outputs."""
    parser = argparse.ArgumentParser(
        description="Build the native Power BI project from Stage 7 CSV exports."
    )
    parser.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    project_file = build_powerbi_project(args.exports, args.output)
    print(f"Power BI project created: {project_file}")
    print("Open the .pbip file in Power BI Desktop and refresh the local CSV sources.")


if __name__ == "__main__":
    main()
