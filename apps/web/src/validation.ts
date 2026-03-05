/** Zod schemas that validate backend payloads and power inferred frontend types. */

import { z } from "zod";

const nonNegativeInt = z.number().int().nonnegative();

export const runIdSchema = z.string().uuid("Invalid run id.");

export const uploadRunResponseSchema = z
  .object({
    run_id: runIdSchema,
  })
  .passthrough();

export const extractResponseSchema = z
  .object({
    run_id: runIdSchema,
    extracted: z.object({
      A: nonNegativeInt,
      B: nonNegativeInt,
    }),
    metrics: z
      .object({
        elapsed_ms: nonNegativeInt,
        docs: z
          .object({
            A: z
              .object({
                primary_model: z.string().min(1).optional(),
                secondary_enabled: z.boolean().optional(),
                model_attempts: nonNegativeInt.optional(),
                model_usage: z.record(z.string(), nonNegativeInt).optional(),
              })
              .passthrough()
              .optional(),
            B: z
              .object({
                primary_model: z.string().min(1).optional(),
                secondary_enabled: z.boolean().optional(),
                model_attempts: nonNegativeInt.optional(),
                model_usage: z.record(z.string(), nonNegativeInt).optional(),
              })
              .passthrough()
              .optional(),
          })
          .passthrough()
          .optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough();

export const mapRoomsResponseSchema = z
  .object({
    run_id: runIdSchema,
    rooms_a: nonNegativeInt,
    rooms_b: nonNegativeInt,
    links: nonNegativeInt,
    metrics: z
      .object({
        elapsed_ms: nonNegativeInt.optional(),
        model_used: z.string().min(1).optional(),
        attempts: nonNegativeInt.optional(),
        candidates_considered: nonNegativeInt.optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough();

export const matchResponseSchema = z
  .object({
    run_id: runIdSchema,
    matches_inserted: nonNegativeInt,
    first_pass_model: z.string().min(1),
    second_pass_models: z.array(z.string().min(1)).min(1),
    first_pass_total_evaluated: nonNegativeInt,
    first_pass_uncertain_count: nonNegativeInt,
    second_pass_reviewed_count: nonNegativeInt,
    second_pass_rooms_invoked: nonNegativeInt,
    nugget_count: nonNegativeInt.optional(),
    status_counts: z.record(z.string(), nonNegativeInt).optional(),
    coverage_audit: z
      .object({
        total_a_items: nonNegativeInt,
        matched_a_rows: nonNegativeInt,
        missing_a_rows: nonNegativeInt,
        critical_blue_count: nonNegativeInt,
        critical_blue_examples: z
          .array(
            z.object({
              item_a_id: nonNegativeInt.optional(),
              room_a: z.string().optional(),
              page: nonNegativeInt.optional(),
              description: z.string().optional(),
            }),
          )
          .optional(),
      })
      .passthrough()
      .optional(),
    llm_telemetry: z
      .object({
        calls: nonNegativeInt.optional(),
        attempts: nonNegativeInt.optional(),
        fallback_successes: nonNegativeInt.optional(),
        reviewer_fallbacks: nonNegativeInt.optional(),
        models_used: z.record(z.string(), nonNegativeInt).optional(),
      })
      .passthrough()
      .optional(),
    elapsed_ms: nonNegativeInt.optional(),
  })
  .passthrough();

export const renderReportResponseSchema = z
  .object({
    run_id: runIdSchema,
    report_path: z.string().min(1),
    metrics: z
      .object({
        elapsed_ms: nonNegativeInt.optional(),
        render_stats: z
          .object({
            line_items_targeted: nonNegativeInt.optional(),
            highlights_added: nonNegativeInt.optional(),
            inline_notes_added: nonNegativeInt.optional(),
            unlocated_notes_added: nonNegativeInt.optional(),
            nugget_summary_notes: nonNegativeInt.optional(),
            critical_blue_summary_notes: nonNegativeInt.optional(),
          })
          .passthrough()
          .optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough();

export const pdfFileSchema = z
  .instanceof(File)
  .refine(
    (file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"),
    "Please select a PDF file.",
  );

export type UploadRunResponse = z.infer<typeof uploadRunResponseSchema>;
export type ExtractResponse = z.infer<typeof extractResponseSchema>;
export type MapRoomsResponse = z.infer<typeof mapRoomsResponseSchema>;
export type MatchResponse = z.infer<typeof matchResponseSchema>;
export type RenderReportResponse = z.infer<typeof renderReportResponseSchema>;
