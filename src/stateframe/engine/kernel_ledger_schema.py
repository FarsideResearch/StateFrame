"""Embedded requirements-ledger schema for offline validation."""

SCHEMA_JSON = r'''
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Source-anchored requirements ledger and fallible checks",
  "description": "Storage schema. Semantic provenance, reference integrity, currentness, authority, and exact quotes require host validation; shape validation is not semantic certification.",
  "type": "object",
  "properties": {
    "schema_version": {
      "const": "requirements-ledger/1"
    },
    "ticket_source_id": {
      "type": "string",
      "minLength": 1
    },
    "current_revision": {
      "type": "string",
      "minLength": 1
    },
    "sources": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Source"
      },
      "minItems": 1
    },
    "spans": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Span"
      },
      "minItems": 1
    },
    "bindings": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Binding"
      },
      "minItems": 0
    },
    "requirements": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Requirement"
      },
      "minItems": 0
    },
    "checks": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Check"
      },
      "minItems": 0
    },
    "certificates": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Certificate"
      },
      "minItems": 0
    },
    "evidence": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Evidence"
      },
      "minItems": 0
    },
    "receipts": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Receipt"
      },
      "minItems": 0
    },
    "disputes": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Dispute"
      },
      "minItems": 0
    },
    "coverage_dispositions": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/CoverageDisposition"
      },
      "minItems": 0
    },
    "policy": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "unknown_blocks_completion": {
          "const": false
        },
        "revalidate_after_source_edits": {
          "const": true
        },
        "max_adjudication_steps": {
          "type": "integer",
          "minimum": 1
        },
        "solver_may_edit_status": {
          "const": false
        },
        "solver_may_edit_checks": {
          "const": false
        }
      },
      "required": [
        "id",
        "unknown_blocks_completion",
        "revalidate_after_source_edits",
        "max_adjudication_steps",
        "solver_may_edit_status",
        "solver_may_edit_checks"
      ],
      "additionalProperties": false
    }
  },
  "required": [
    "schema_version",
    "ticket_source_id",
    "current_revision",
    "sources",
    "spans",
    "bindings",
    "requirements",
    "checks",
    "certificates",
    "evidence",
    "receipts",
    "disputes",
    "coverage_dispositions",
    "policy"
  ],
  "additionalProperties": false,
  "$defs": {
    "VersionRef": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "version": {
          "type": "integer",
          "minimum": 1
        }
      },
      "required": [
        "id",
        "version"
      ],
      "additionalProperties": false
    },
    "Source": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "kind": {
          "enum": [
            "ticket",
            "public_source",
            "public_test",
            "public_documentation"
          ]
        },
        "path": {
          "type": "string",
          "minLength": 1
        },
        "revision": {
          "type": "string",
          "minLength": 1
        },
        "sha256": {
          "type": "string",
          "pattern": "^[0-9a-f]{64}$"
        },
        "encoding": {
          "const": "utf-8"
        },
        "text": {
          "type": "string"
        }
      },
      "required": [
        "id",
        "kind",
        "path",
        "revision",
        "sha256",
        "encoding",
        "text"
      ],
      "additionalProperties": false
    },
    "Span": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "source_id": {
          "type": "string",
          "minLength": 1
        },
        "source_sha256": {
          "type": "string",
          "pattern": "^[0-9a-f]{64}$"
        },
        "start_byte": {
          "type": "integer",
          "minimum": 0
        },
        "end_byte": {
          "type": "integer",
          "minimum": 0
        },
        "quote": {
          "type": "string",
          "minLength": 1
        },
        "context_span_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        }
      },
      "required": [
        "id",
        "source_id",
        "source_sha256",
        "start_byte",
        "end_byte",
        "quote",
        "context_span_ids"
      ],
      "additionalProperties": false,
      "description": "Byte offsets are half-open in the immutable UTF-8 source. The host must verify the exact slice; JSON Schema alone cannot do so."
    },
    "Locator": {
      "type": "object",
      "properties": {
        "source_id": {
          "type": "string",
          "minLength": 1
        },
        "source_sha256": {
          "type": "string",
          "pattern": "^[0-9a-f]{64}$"
        },
        "revision": {
          "type": "string",
          "minLength": 1
        },
        "path": {
          "type": "string",
          "minLength": 1
        },
        "start_line": {
          "type": "integer",
          "minimum": 1
        },
        "end_line": {
          "type": "integer",
          "minimum": 1
        },
        "symbol": {
          "type": [
            "string",
            "null"
          ]
        }
      },
      "required": [
        "source_id",
        "source_sha256",
        "revision",
        "path",
        "start_line",
        "end_line",
        "symbol"
      ],
      "additionalProperties": false
    },
    "Binding": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "kind": {
          "enum": [
            "public_symbol",
            "http_endpoint",
            "cli_surface",
            "configuration_key",
            "artifact"
          ]
        },
        "name": {
          "type": "string",
          "minLength": 1
        },
        "role": {
          "enum": [
            "entrypoint",
            "input_field",
            "output_field",
            "explicitly_required_interface",
            "other"
          ]
        },
        "status": {
          "enum": [
            "resolved",
            "explicit_new_interface",
            "unresolved",
            "ambiguous"
          ]
        },
        "locator": {
          "anyOf": [
            {
              "$ref": "#/$defs/Locator"
            },
            {
              "type": "null"
            }
          ]
        },
        "support_span_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        },
        "public_usage_evidence_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        }
      },
      "required": [
        "id",
        "kind",
        "name",
        "role",
        "status",
        "locator",
        "support_span_ids",
        "public_usage_evidence_ids"
      ],
      "additionalProperties": false
    },
    "Predicate": {
      "type": "object",
      "properties": {
        "observable": {
          "type": "string",
          "minLength": 1
        },
        "operator": {
          "enum": [
            "equals",
            "not_equals",
            "less_than",
            "less_or_equal",
            "greater_than",
            "greater_or_equal",
            "member_of",
            "absent",
            "present",
            "instance_of",
            "unchanged",
            "relation"
          ]
        },
        "expected": {},
        "support_span_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        }
      },
      "required": [
        "observable",
        "operator",
        "expected",
        "support_span_ids"
      ],
      "additionalProperties": false,
      "description": "Operator semantics must be implemented by a versioned host template, not arbitrary model text. Every expected value and observation path needs provenance."
    },
    "Contract": {
      "type": "object",
      "properties": {
        "modality": {
          "enum": [
            "required",
            "prohibited",
            "unresolved"
          ]
        },
        "subject_binding_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        },
        "preconditions": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Predicate"
          },
          "minItems": 0
        },
        "action": {
          "type": "string",
          "minLength": 1
        },
        "expectations": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Predicate"
          },
          "minItems": 1
        },
        "quantifier": {
          "enum": [
            "universal_in_stated_scope",
            "existential",
            "finite_enumerated",
            "unresolved"
          ]
        },
        "scope": {
          "type": "string",
          "minLength": 1
        },
        "exception_span_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "unresolved_interpretations": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "derivation_template": {
          "type": [
            "string",
            "null"
          ]
        }
      },
      "required": [
        "modality",
        "subject_binding_ids",
        "preconditions",
        "action",
        "expectations",
        "quantifier",
        "scope",
        "exception_span_ids",
        "unresolved_interpretations",
        "derivation_template"
      ],
      "additionalProperties": false
    },
    "Assessment": {
      "type": "object",
      "properties": {
        "status": {
          "enum": [
            "satisfied",
            "open",
            "counterexample"
          ]
        },
        "at_revision": {
          "type": "string",
          "minLength": 1
        },
        "reason": {
          "type": "string",
          "minLength": 1
        },
        "basis": {
          "enum": [
            "none",
            "acceptance_policy",
            "exhaustive",
            "formal"
          ]
        },
        "policy_ref": {
          "type": [
            "string",
            "null"
          ]
        },
        "evidence_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "scope": {
          "type": "string",
          "minLength": 1
        },
        "limitations": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        }
      },
      "required": [
        "status",
        "at_revision",
        "reason",
        "basis",
        "policy_ref",
        "evidence_ids",
        "scope",
        "limitations"
      ],
      "additionalProperties": false,
      "description": "Host-reduced projection; never accept a solver-authored status. Satisfied means only the stated evidence scope, unless exhaustive/formal sufficiency is established."
    },
    "Requirement": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "version": {
          "type": "integer",
          "minimum": 1
        },
        "kind": {
          "enum": [
            "behavior",
            "error_behavior",
            "interface",
            "compatibility",
            "representation",
            "artifact",
            "nonfunctional",
            "unresolved"
          ]
        },
        "source_span_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        },
        "exact_clause": {
          "type": "string",
          "minLength": 1
        },
        "contract": {
          "$ref": "#/$defs/Contract"
        },
        "check_refs": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/VersionRef"
          },
          "minItems": 0
        },
        "assessment": {
          "$ref": "#/$defs/Assessment"
        },
        "supersedes": {
          "anyOf": [
            {
              "$ref": "#/$defs/VersionRef"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "id",
        "version",
        "kind",
        "source_span_ids",
        "exact_clause",
        "contract",
        "check_refs",
        "assessment",
        "supersedes"
      ],
      "additionalProperties": false
    },
    "Certificate": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "stage": {
          "enum": [
            "source_alignment",
            "binding",
            "fixture",
            "oracle",
            "runner"
          ]
        },
        "verdict": {
          "enum": [
            "supported",
            "unsupported",
            "unknown"
          ]
        },
        "method": {
          "enum": [
            "deterministic_template",
            "public_fact_validation",
            "independent_semantic_review",
            "formal_derivation"
          ]
        },
        "producer": {
          "type": "string",
          "minLength": 1
        },
        "producer_version": {
          "type": "string",
          "minLength": 1
        },
        "evidence_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        },
        "normative_assumptions": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "limitations": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        }
      },
      "required": [
        "id",
        "stage",
        "verdict",
        "method",
        "producer",
        "producer_version",
        "evidence_ids",
        "normative_assumptions",
        "limitations"
      ],
      "additionalProperties": false,
      "description": "Supported is a policy decision with evidence, not a theorem unless the method and proof establish one."
    },
    "Authorization": {
      "type": "object",
      "properties": {
        "kind": {
          "enum": [
            "calibrated_template_family",
            "formal_derivation",
            "explicit_public_acceptance_test"
          ]
        },
        "policy_ref": {
          "type": "string",
          "minLength": 1
        },
        "evidence_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        }
      },
      "required": [
        "kind",
        "policy_ref",
        "evidence_ids"
      ],
      "additionalProperties": false
    },
    "Admission": {
      "type": "object",
      "properties": {
        "tier": {
          "enum": [
            "rejected",
            "advisory",
            "shadow",
            "gating"
          ]
        },
        "reason": {
          "type": "string",
          "minLength": 1
        },
        "certificate_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "authorization": {
          "anyOf": [
            {
              "$ref": "#/$defs/Authorization"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "tier",
        "reason",
        "certificate_ids",
        "authorization"
      ],
      "additionalProperties": false,
      "allOf": [
        {
          "if": {
            "properties": {
              "tier": {
                "const": "gating"
              }
            },
            "required": [
              "tier"
            ]
          },
          "then": {
            "properties": {
              "authorization": {
                "$ref": "#/$defs/Authorization"
              }
            },
            "required": [
              "authorization"
            ]
          }
        }
      ]
    },
    "Fixture": {
      "type": "object",
      "properties": {
        "input": {},
        "setup_evidence_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        },
        "varied_dimensions": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "validated_precondition_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "reset_strategy": {
          "type": "string",
          "minLength": 1
        },
        "environment_ref": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "input",
        "setup_evidence_ids",
        "varied_dimensions",
        "validated_precondition_ids",
        "reset_strategy",
        "environment_ref"
      ],
      "additionalProperties": false
    },
    "Oracle": {
      "type": "object",
      "properties": {
        "template_id": {
          "type": "string",
          "minLength": 1
        },
        "assertions": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Predicate"
          },
          "minItems": 1
        },
        "scope": {
          "type": "string",
          "minLength": 1
        },
        "unsupported_assertions": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        }
      },
      "required": [
        "template_id",
        "assertions",
        "scope",
        "unsupported_assertions"
      ],
      "additionalProperties": false
    },
    "Check": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "version": {
          "type": "integer",
          "minimum": 1
        },
        "requirement_refs": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/VersionRef"
          },
          "minItems": 1
        },
        "lifecycle": {
          "enum": [
            "active",
            "superseded",
            "retired"
          ]
        },
        "purpose": {
          "enum": [
            "change_detection",
            "preservation",
            "mixed",
            "inspection"
          ]
        },
        "source_span_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        },
        "binding_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        },
        "fixture": {
          "$ref": "#/$defs/Fixture"
        },
        "oracle": {
          "$ref": "#/$defs/Oracle"
        },
        "artifact_path": {
          "type": "string",
          "minLength": 1
        },
        "artifact_sha256": {
          "type": "string",
          "pattern": "^[0-9a-f]{64}$"
        },
        "command": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        },
        "generator": {
          "type": "object",
          "properties": {
            "kind": {
              "enum": [
                "deterministic_host",
                "proposal_model",
                "public_test"
              ]
            },
            "id": {
              "type": "string",
              "minLength": 1
            },
            "version": {
              "type": "string",
              "minLength": 1
            },
            "input_source_ids": {
              "type": "array",
              "items": {
                "type": "string",
                "minLength": 1
              },
              "minItems": 1
            }
          },
          "required": [
            "kind",
            "id",
            "version",
            "input_source_ids"
          ],
          "additionalProperties": false
        },
        "admission": {
          "$ref": "#/$defs/Admission"
        },
        "base_receipt_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "current_receipt_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "supersedes": {
          "anyOf": [
            {
              "$ref": "#/$defs/VersionRef"
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "id",
        "version",
        "requirement_refs",
        "lifecycle",
        "purpose",
        "source_span_ids",
        "binding_ids",
        "fixture",
        "oracle",
        "artifact_path",
        "artifact_sha256",
        "command",
        "generator",
        "admission",
        "base_receipt_ids",
        "current_receipt_ids",
        "supersedes"
      ],
      "additionalProperties": false
    },
    "Evidence": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "kind": {
          "enum": [
            "source_binding",
            "fixture_validation",
            "execution",
            "trace",
            "inspection",
            "formal_proof",
            "calibration"
          ]
        },
        "producer": {
          "const": "host"
        },
        "revision": {
          "type": "string",
          "minLength": 1
        },
        "locator": {
          "anyOf": [
            {
              "$ref": "#/$defs/Locator"
            },
            {
              "type": "null"
            }
          ]
        },
        "span_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "receipt_id": {
          "type": [
            "string",
            "null"
          ]
        },
        "claim": {
          "type": "string",
          "minLength": 1
        },
        "scope": {
          "type": "string",
          "minLength": 1
        },
        "artifact_ref": {
          "type": [
            "string",
            "null"
          ]
        }
      },
      "required": [
        "id",
        "kind",
        "producer",
        "revision",
        "locator",
        "span_ids",
        "receipt_id",
        "claim",
        "scope",
        "artifact_ref"
      ],
      "additionalProperties": false
    },
    "Receipt": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "producer": {
          "const": "host"
        },
        "check_ref": {
          "$ref": "#/$defs/VersionRef"
        },
        "phase": {
          "enum": [
            "base",
            "candidate",
            "control"
          ]
        },
        "repository_revision": {
          "type": "string",
          "minLength": 1
        },
        "artifact_sha256": {
          "type": "string",
          "pattern": "^[0-9a-f]{64}$"
        },
        "environment_ref": {
          "type": "string",
          "minLength": 1
        },
        "command": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1
        },
        "outcome": {
          "enum": [
            "pass",
            "assertion_failure",
            "error",
            "skip",
            "not_applicable"
          ]
        },
        "assertion_reached": {
          "type": "boolean"
        },
        "preconditions_validated": {
          "type": "boolean"
        },
        "attribution": {
          "enum": [
            "target_assertion",
            "other_requirement",
            "harness",
            "unknown"
          ]
        },
        "observed": {},
        "expected": {},
        "trace_ref": {
          "type": [
            "string",
            "null"
          ]
        },
        "determinism": {
          "enum": [
            "reproduced",
            "single_run",
            "flaky",
            "not_applicable"
          ]
        }
      },
      "required": [
        "id",
        "producer",
        "check_ref",
        "phase",
        "repository_revision",
        "artifact_sha256",
        "environment_ref",
        "command",
        "outcome",
        "assertion_reached",
        "preconditions_validated",
        "attribution",
        "observed",
        "expected",
        "trace_ref",
        "determinism"
      ],
      "additionalProperties": false
    },
    "Dispute": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "minLength": 1
        },
        "target_check_ref": {
          "$ref": "#/$defs/VersionRef"
        },
        "raised_at_revision": {
          "type": "string",
          "minLength": 1
        },
        "ground": {
          "enum": [
            "not_entailed",
            "wrong_binding",
            "invalid_fixture",
            "wrong_observable",
            "harness_error",
            "stale_evidence",
            "conflicting_source"
          ]
        },
        "claim": {
          "type": "string",
          "minLength": 1
        },
        "source_span_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "evidence_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "reproduction_command": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 0
        },
        "alternative_interpretation": {
          "type": [
            "string",
            "null"
          ]
        },
        "resolution": {
          "type": "object",
          "properties": {
            "outcome": {
              "enum": [
                "pending",
                "upheld",
                "check_invalid",
                "check_rebound",
                "interpretation_open"
              ]
            },
            "adjudicator": {
              "type": [
                "string",
                "null"
              ]
            },
            "reason": {
              "type": [
                "string",
                "null"
              ]
            },
            "evidence_ids": {
              "type": "array",
              "items": {
                "type": "string",
                "minLength": 1
              },
              "minItems": 0
            },
            "replacement_check_ref": {
              "anyOf": [
                {
                  "$ref": "#/$defs/VersionRef"
                },
                {
                  "type": "null"
                }
              ]
            }
          },
          "required": [
            "outcome",
            "adjudicator",
            "reason",
            "evidence_ids",
            "replacement_check_ref"
          ],
          "additionalProperties": false
        }
      },
      "required": [
        "id",
        "target_check_ref",
        "raised_at_revision",
        "ground",
        "claim",
        "source_span_ids",
        "evidence_ids",
        "reproduction_command",
        "alternative_interpretation",
        "resolution"
      ],
      "additionalProperties": false
    },
    "CoverageDisposition": {
      "type": "object",
      "properties": {
        "span_id": {
          "type": "string",
          "minLength": 1
        },
        "disposition": {
          "enum": [
            "mapped",
            "context_only",
            "non_normative",
            "unresolved"
          ]
        },
        "requirement_refs": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/VersionRef"
          },
          "minItems": 0
        },
        "reason": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "span_id",
        "disposition",
        "requirement_refs",
        "reason"
      ],
      "additionalProperties": false
    }
  }
}
'''
