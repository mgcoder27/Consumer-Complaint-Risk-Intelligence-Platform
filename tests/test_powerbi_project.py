"""Tests for generated Power BI project structure and report field bindings."""

import json
import tempfile
import unittest
from pathlib import Path

from src.build_powerbi_project import (
    MODEL_NAME,
    REPORT_NAME,
    TABLES,
    build_powerbi_project,
)


class PowerBIProjectTests(unittest.TestCase):
    """Verify PBIP manifests and every visual-to-model field reference."""

    def test_project_contains_four_pages_and_valid_field_references(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        exports = project_root / "data" / "processed" / "powerbi"

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "powerbi"
            project_file = build_powerbi_project(exports, output_dir)

            project_manifest = json.loads(project_file.read_text(encoding="utf-8"))
            report_dir = output_dir / REPORT_NAME
            report_definition = json.loads(
                (report_dir / "definition" / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            page_manifest = json.loads(
                (
                    report_dir
                    / "definition"
                    / "pages"
                    / "pages.json"
                ).read_text(encoding="utf-8")
            )
            visual_files = list(
                (report_dir / "definition" / "pages").glob(
                    "*/visuals/*/visual.json"
                )
            )

            self.assertTrue(
                (output_dir / MODEL_NAME / "definition" / "expressions.tmdl").is_file()
            )
            self.assertEqual(len(page_manifest["pageOrder"]), 4)
            self.assertEqual(page_manifest["activePageName"], "OverviewPage")
            self.assertIn("themeCollection", report_definition)
            self.assertEqual(len(visual_files), 14)
            self.assertEqual(len(project_manifest["artifacts"]), 2)

            for visual_path in visual_files:
                visual_data = json.loads(visual_path.read_text(encoding="utf-8"))
                visual_type = visual_data["visual"]["visualType"]
                self.assertIn(
                    visual_type,
                    {"card", "lineChart", "clusteredBarChart", "tableEx"},
                )
                if visual_type == "lineChart":
                    sort_specification = visual_data["visual"]["query"][
                        "sortDefinition"
                    ]["sort"][0]
                    self.assertEqual(
                        sort_specification["field"]["Column"]["Property"],
                        "month",
                    )
                    self.assertEqual(sort_specification["direction"], "Ascending")
                query_state = visual_data["visual"]["query"]["queryState"]
                for role in query_state.values():
                    for projection in role["projections"]:
                        field = projection["field"]
                        field_kind = "Measure" if "Measure" in field else "Column"
                        model_field = field[field_kind]
                        entity = model_field["Expression"]["SourceRef"]["Entity"]
                        property_name = model_field["Property"]
                        self.assertIn(entity, TABLES)
                        if field_kind == "Measure":
                            self.assertIn(property_name, TABLES[entity]["measures"])
                        else:
                            self.assertIn(property_name, TABLES[entity]["columns"])

    def test_missing_stage_seven_exports_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "Generate Stage 7 exports"):
                build_powerbi_project(
                    Path(temporary_directory),
                    Path(temporary_directory) / "powerbi",
                )


if __name__ == "__main__":
    unittest.main()
