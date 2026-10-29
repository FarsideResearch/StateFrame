"""Feature configuration grouped into one dataclass per component.

GymConfig collects the settings passed to each runner. Result rows record the
configuration fingerprint so behavior can be compared across runs.
"""
from dataclasses import dataclass, field, asdict
import hashlib, json, os


@dataclass
class LocateCfg:
    enabled: bool = True
    bm25_top: int = 24              # lexical survivors before union
    prox_seed: int = 8              # top-N whose import neighbors join the pool
    pool_cap: int = 48              # relevance-atom input size (widened per
                                    # a_disjoint_locate: inference is free)
    menu_k: int = 8                 # final file menu size
    # POOL-RECALL WIDENING (/tmp/solq/w50/a_pool_recall.md, wired 2026-08-29).
    # Two-stage: BM25 top-W, keep candidates with >= min_hits distinct symptom
    # tokens, rerank, cap. Its paths are APPENDED after the existing menu
    # prefix and can only reach the overflow menu, so the primary menu (and
    # therefore the false-locate rate) is provably unchanged. The four known
    # pool misses at MC scale sit at Stage-A ranks 19-58: inside W=100, outside
    # today's cap of 48.
    pool_recall: bool = True
    pool_recall_w: int = 100        # Stage-A sweep width
    pool_recall_min_hits: int = 2   # the precision gate; gold always clears it
    pool_recall_cap: int = 64       # covers rank 58 with a small margin
    # CONSULTATION CHANNELS (wired 2026-08-29). Ranking inputs only: bounded,
    # permutation-preserving, hard anchors pinned, empty bundle == baseline.
    # `case_learning` is ON and currently INERT by construction -- no learned
    # case has been mined yet (results/case_learning.jsonl does not exist), so
    # its bundle is empty and the record reads `neutral`. That is the honest
    # state: instrumented, live, and measurably doing nothing until cases
    # exist. `blame_channel` needs a real git walk per task; it is ON and
    # fails open to `unavailable` on a corpus that is not a git checkout.
    blame_channel: bool = True
    blame_window: int = 200
    blame_baseline_window: int = 500
    case_learning: bool = True
    # LOCATE RERANK COMPOSITION. All ranking adjusters propose against the
    # same incoming order; `joint` commits one budgeted permutation and `off`
    # keeps that order while retaining proposal/evidence telemetry.
    rerank_compose: str = "joint"
    rerank_compose_budget: int = 2
    relevance_atoms: bool = True    # 1B ternary filter (reject confident NOs only)
    evidence_cards: bool = True     # show matched identifiers per candidate
    anchors: bool = True            # exact-symbol stage (fn > class)
    anchor_forces_file: bool = True # unique fn-anchor skips the menu entirely
    require_file_support: bool = True
                                    # A MENU file pick must name its evidence:
                                    # a symptom anchor, a task code-token the
                                    # file contains, or DISCRIMINATING structural
                                    # feasibility. Real-19 autopsy fix #3: with a
                                    # scrubbed symptom the menu collapsed onto
                                    # the same two attractor files for five
                                    # unrelated tasks, unanimously and at margins
                                    # up to 2.82. ON: this turns a confident
                                    # fabrication into an honest abstain. Pinned
                                    # picks (plan/anchor) are never gated.
    file_support_content_min: int = 2
                                    # GROUNDED-SYMPTOM support channel: how many
                                    # DISTINCTIVE symptom content words a file
                                    # must carry for the pick to count as
                                    # evidenced. 0 = channel off.
                                    # Measured 2026-08-29 on the fresh-21 real
                                    # board (offline, per-task): the support gate
                                    # abstained 21/21 while the model's pick was
                                    # already the gold file 16/21 — the two
                                    # existing channels (snake_case content
                                    # tokens; structural class feasibility) are
                                    # both unfireable on prose tickets whose
                                    # class is a census shape. Distinctive-word
                                    # counts, gold vs the model's actual pick:
                                    #   correct picks (16): 2,2,3,3,3,3,4,4,4,4,
                                    #                       5,5,5,6,7,7
                                    #   confabulated (5):   1,1,1,1,2
                                    # So 2 admits every correct pick and still
                                    # refuses 4 of 5 confabulations (all four are
                                    # utils/attempt_context.py, the real-19
                                    # attractor this gate exists for). 3 would
                                    # cost 2 correct picks to buy 1 refusal.
    function_granularity: bool = True  # func_index dual-view retrieval (+4 menu
                                       # recall offline); file view stays fallback
    dunder_expansion: bool = False  # SHIPS OFF ON MEASURED EVIDENCE (rung-4000
                                    # autopsy, ranked fix #6b — the fix was
                                    # BUILT, MEASURED and REFUSED, not skipped).
                                    #
                                    # The C7 hole is real: models/learner_concept
                                    # _gap.py::__repr__ has never reached a file
                                    # menu, two cycles running, on the ticket "A
                                    # function responsible for 'repr'". The
                                    # autopsy proposed two moves. Move (a),
                                    # "confirm models/ is in the corpus at all",
                                    # is ANSWERED: it is — load_repo collects 35
                                    # models/*.py of 210 files and the gold file
                                    # is among them, so a whole-directory
                                    # absence is NOT the explanation.
                                    #
                                    # Move (b), expanding the dunder into its
                                    # lexical form, was then measured against the
                                    # live corpus (gold file's rank in the fused
                                    # function ranking, lower is better):
                                    #   baseline                        45
                                    #   "repr representation string
                                    #    display format debug"         117  WORSE
                                    #   "__repr__"                      45  no-op
                                    #   "__repr__ repr"                 45  no-op
                                    #   "__repr__ __repr__ __repr__"    45  no-op
                                    # The synonyms are ordinary English and flood
                                    # the ranking toward prose-heavy files; the
                                    # dunder spellings are no-ops because the
                                    # tokenizer splits on underscores and drops
                                    # tokens under three characters, so
                                    # `__repr__` reduces to `repr`, which the
                                    # query already contained. Neither moves the
                                    # gold file one rank closer to the menu.
                                    #
                                    # So the mechanism ships knob-routed, tested,
                                    # and OFF: turning it on would cost menu
                                    # quality on every OTHER task to buy nothing
                                    # on this one. The C7 lever is elsewhere —
                                    # anchor_recall_k (which already reaches this
                                    # file at rank 0 of its own anchor set) or
                                    # the `insufficient_menu` escape the model
                                    # asked for in its own trace.
                                    #
                                    # When on, the expansion is APPENDED to the
                                    # ranking query only, never to a prompt.
    docstring_function_pin: bool = True  # a function whose docstring the task
                                    # quotes VERBATIM wins the function stage
                                    # outright (rung-4000 autopsy, ranked fix
                                    # #7). MEASURED: api/teacher_chat.py voted
                                    # unanimously for `_legacy_chat_stream`
                                    # because the symptom-word ranker counted 6
                                    # words there against 5 in `create_session`
                                    # — while the ticket quotes create_session's
                                    # docstring verbatim. A verbatim docstring is
                                    # near-certain provenance and must
                                    # short-circuit a word COUNT, the same way
                                    # anchor_forces_file already short-circuits
                                    # the file menu.
    recall_append_k: int = 0        # APPEND-ONLY recall arm: N construct-union
                                    # pool files appended AFTER the ranked menu
                                    # (never reordering it) and asked as a SECOND
                                    # menu only when the primary produces no pick.
                                    # It does raise reachability (gold file in the
                                    # menu 16/21 -> 21/21) but the live 08-28
                                    # head-to-head says that is not where the
                                    # value is: the overflow menu fired 3 times,
                                    # was wrong 3 times, and turned honest
                                    # ABSTAINs into wrong picks — mechanical
                                    # location failures 6 with it OFF vs 9 ON.
                                    # Ships OFF on the evidence; the mechanism
                                    # stays knob-routed and tested.
    function_menu_primary_k: int = 8  # THE FUNCTION MENU'S PRIMARY/OVERFLOW SPLIT,
                                    # ported from the file menu (E100 task 2).
                                    # MEASURED BLOWER, 252c12de: the function menu
                                    # offered all 26 candidates at once, the prompt
                                    # reached 7596 tokens and 3/3 calls came back
                                    # finish_reason='length'. The harness refused
                                    # the truncated completions, which is #11d
                                    # working exactly as designed — and also three
                                    # runs that could never produce an answer.
                                    # The file menu solved this shape already:
                                    # bound what the model is asked in ONE go and
                                    # make the rest a SECOND menu, asked only when
                                    # the first produced no pick. Same width budget
                                    # (MAX_MODEL_FILE_MENU_OPTIONS) for the same
                                    # measured reason: a wider menu buys recall and
                                    # pays in no-picks. Under the binary selector
                                    # it also bounds the probe BILL, since every
                                    # extra primary option is another call.
                                    # 0 disables the split — the 26-option menu is
                                    # then byte-identical to the pre-split path,
                                    # which is what the letter byte-identity test
                                    # pins.
    anchor_recall_k: int = 2        # APPEND-ONLY anchor recall arm (E52): when a
                                    # quoted symptom phrase matches MANY function
                                    # names the pin is refused and the whole
                                    # anchor set used to be discarded. Its top-K
                                    # paths by fused score are now APPENDED after
                                    # everything retrieval produced, so they can
                                    # only ever reach the OVERFLOW menu. K is
                                    # deliberately tiny: the construct-union
                                    # overflow arm (recall_append_k) measured
                                    # NEGATIVE at width 4 because a wide second
                                    # menu converts honest ABSTAINs into wrong
                                    # picks. This arm is exact-name evidence and
                                    # the gold sat at rank 0 of its own set.
    symptom_anchors: bool = True    # quoted symptom phrase whose words equal a
                                    # function name -> pin file+function when the
                                    # match is unique (4/4 correct, 3 unique/21)
    rank_query_facts: bool = False  # does the FACTS block (class attribute lists)
                                    # join the RANKING query? It is written for
                                    # the deciding model; as query terms it
                                    # reranks the corpus. Prompts always show it.
                                    # Measured 08-28 over the 21-task gym through
                                    # the real pipeline: True 20/21 menu recall,
                                    # False 21/21 (a dozen injected attribute
                                    # words outvoted the symptom's own words).
    function_rank_by_evidence: bool = True  # order the FUNCTION menu by the
                                    # func_index fused score instead of AST-walk
                                    # order. Measured 08-28: gold is option A in
                                    # 19/21 vs 4/21, top-3 in 21/21.
    function_evidence: bool = True  # per-option evidence lines on the function
                                     # menu (matched symptom words, construct
                                     # count, size) instead of bare names
    function_support_gate: bool = True  # deterministic pre-authoring check:
                                      # reject a menu pick only when another
                                      # offered function has materially stronger
                                      # code-grounded support; lexical absence
                                      # alone remains inconclusive.
    file_rank_fusion: bool = True   # E51: reorder the FILE menu by an explicit
                                    # fusion of signals the pipeline ALREADY
                                    # computes. STRICT PERMUTATION of the menu
                                    # the old code produced — membership, width
                                    # and the overflow split are untouched, so
                                    # menu recall is provably unchanged and the
                                    # only thing that can move is position.
                                    # WHY (measured 2026-08-29 over the 21-task
                                    # gym, 3 cycles, per-task keyed): the 1B
                                    # picked option A in 15/16 file menus. Gold
                                    # at idx0 -> 8/10 correct; gold anywhere
                                    # else -> 0/6. File-menu accuracy IS ranker
                                    # top-1 accuracy (the E49 mutation-menu
                                    # finding, one stage upstream). Offline
                                    # replay of all 20 synthetic tasks (0 model
                                     # calls): gold@0 10/20 -> 16/20, i.e. 16/16
                                     # of the menus that contain gold, with 6
                                     # conversions and 0 regressions.
    file_rank_window: int = 64       # 64 = ON by FULL-CORPUS measurement (E142,
                                     # 2026-08-30, /tmp/FANIN_AB.md): swept over
                                     # all 21 real tasks, window 64 improves gold
                                     # rank on 2 and worsens 0 — gold-at-rank-1
                                     # goes 9 -> 11. Window 128 is equivalent.
                                     #
                                     # SUPERSEDES the earlier `0` setting, which
                                     # was chosen from a SINGLE canary task
                                     # (ea639fc1) whose menu lost gold at 64. That
                                     # one case did not generalise: it cost us two
                                     # tasks corpus-wide for as long as it was
                                     # live. One task is not a corpus — knob
                                     # decisions come from a full-bench sweep,
                                     # which is deterministic and takes seconds.
                                     # Re-enable only with a canary-green replay.
    file_rank_lennorm_b: float = 0.45   # BM25 length normalization, applied at
                                    # the FILE level. func_index._bm25 has no
                                    # b/avgdl term at all, so raw term frequency
                                    # wins and the biggest module in the corpus
                                    # (api/lessons.py, whose winning "function"
                                    # was the <module> pseudo-record) took idx0
                                    # in 6 of the 6 wrong-at-zero menus.
                                    # 0 = off, 1 = full. Measured flat plateau
                                    # 0.35-0.55 at every content weight >= 0.10;
                                    # 0.45 is its middle.
    file_rank_content_w: float = 0.20   # reward per matched CONTENT word (the
                                      # ticket's own vocabulary minus the words
                                     # every bug ticket contains: "find the
                                     # function responsible for ... fix the
                                     # bug"). Same defect as rank_query_facts:
                                    # boilerplate outvoting symptom words. The
                                    # losers matched on {and, bug, fix, the};
                                    # the gold files matched on {files, number,
                                     # submission, validate}. Flat plateau
                                      # 0.10-0.30; 0.20 is its middle.
    file_rank_template_stoplist: bool = True  # remove fixed prose from the
                                       # grounded ticket template before content
                                       # scoring; old-ranker replay sets false.
    file_rank_definition_w: float = 4.0  # bounded bonus for symptom tokens
                                       # defined by the file's AST; measured
                                       # 2026-08-29 as the strongest safe arm.
    file_rank_definition_min: float = 0.0  # minimum AST definition coverage for
                                      # that bonus; zero means every coverage
                                      # value is eligible.
    file_rank_definition_guard: bool = True  # raw definition arm retains the
                                       # incumbent rank-1 file in the eight-file
                                       # primary menu; dormant while window=0.
    # ------------------------------------------------ FAN-IN ANTI-ATTRACTOR
    # THE 7 ABSTAINS (w38 `a_abstain_s.md`, w42 `a_abstain_replay.md`).
    # Measured on the fresh-21 real board: six tasks abstained and one picked
    # the wrong file, and all six abstains died at the file-support gate on the
    # SAME file — `utils/attempt_context.py`, the real-19 attractor — for six
    # unrelated tickets. A scrubbed symptom carries no content tokens, so the
    # ranker falls back on the corpus's own term mass and collapses onto hubs.
    #
    # The prior: a file that more than `fan_in_k` other application files import
    # is a hub, and a hub's fused score is multiplied by `fan_in_m` (< 1). Never
    # an exclusion, never applied to a hard anchor, applied once.
    #
    # SELECTED BY REPLAY. `probes/locate_fan_in_replay.py` grids K x m over the
    # 21 with the objective "maximise gold-top1 subject to ZERO new
    # false-locates on the currently-correct tasks"; the guard test
    # `test_locate_fan_in_offline.py` pins the selected pair against a fixture digest so
    # it can never be re-discovered at runtime. Measured grid:
    #
    #
    #        K     m   gold_top1  new_false  feasible
    #        5  0.50/0.70/0.85     7/7/8      2/2/1      NO
    #       10  0.50/0.70          9/9        1/1        NO
    #       10  0.85              10          0          yes
    #       20  0.50/0.70/0.85    10/10/10    0/0/0      yes   <- selected
    #       40  0.50/0.70/0.85     9/9/9      0/0/0      yes
    #
    # Baseline is 9. K=20/m=0.85 is chosen as the LEAST AGGRESSIVE feasible pair
    # at the maximum (largest K, m nearest 1.0): K=10/m=0.85 scores the same 10
    # while penalising strictly more files, so it buys nothing and enlarges the
    # known failure mode for free.
    #
    # HONEST CEILING, because the headline number invites the wrong reading:
    # this converts exactly ONE of the seven the gym could not locate (t17). It
    # is not the fix for the other six and does not claim to be — four of them
    # (t1, t6, t14, t15) are POOL-RECALL misses whose gold never reaches the
    # eight-option menu at all, and no reordering prior can move a candidate
    # that is not in the list being reordered.
    #
    # KNOWN FAILURE MODE, unchanged: when the bug genuinely IS in the shared
    # utility this demotes the gold by design. Bounded by the anchor exemption,
    # the single multiplier, and the zero-regression gate the pair had to clear.
    # OWNERSHIP TIE-BREAKERS (`locate-attractor-tie-breakers.md`, w33). The
    # fuller model the fan-in prior alone could not deliver: an additive
    # S = 60B + 15D + 5U + 12V - 10*H*(1-D)^2 over definition-site match (D),
    # use-site match (U), symptom-verb/file-role compatibility (V) and damped
    # import hubness (H). Reorders NON-ANCHORED candidates only; never changes
    # menu membership, so pool recall is untouched by construction.
    # BUILT, MEASURED, AND REFUSED AS A DEFAULT — the same disposition, and the
    # same kind of evidence, as `dunder_expansion` and `typed_holes`.
    # `probes/locate_fan_in_replay.py --arms`, faithful per-repo corpus, 21 rows:
    #
    #     baseline           gold_top1 =  9   new_false = []
    #     fan_in_only        gold_top1 = 10   new_false = []      <- shipped
    #     ownership_only     gold_top1 =  8   new_false = [3]
    #     ownership+fan_in   gold_top1 =  9   new_false = [3]
    #
    # (Measured AFTER `symptom_vocabulary` landed. Before the grounded-noun
    # extractor the ownership arms scored 9 and 10 with no regression — i.e.
    # giving the ownership model a CLEANER symptom vocabulary made it WORSE, on
    # 21 rows. That is a small sample and the honest reading is "no signal",
    # not "cleaner input hurts"; it is recorded because a number that moved the
    # wrong way after an improvement is exactly the kind of thing that gets
    # quietly dropped.)
    #
    # The full ownership model COSTS a row on its own and adds nothing on top of
    # fan-in. Its features are real but they do not discriminate on this board:
    # measured on t2/t5/t17 the role term V is 0.50 for EVERY candidate (the
    # grounded-symptom template contains no verb the role map recognises), so
    # the 12V term is a constant that cannot reorder anything, and D/U are all
    # under 0.3 with the gold not reliably highest. Shipping it would spend a
    # real failure mode (a genuine bug in a hub gets demoted) for a measured
    # negative. It stays here, off, with its numbers, because "we tried the
    # fuller design and it lost" is a result worth keeping.
    ownership_rank: bool = False
    # SHIPS ON at the replay-selected pair. Feasible (zero new false-locates on
    # the currently-correct set) and the only arm that beat baseline.
    fan_in_penalty: bool = True
    fan_in_k: int = 20              # distinct in-corpus importers; `== k` is
                                    # NOT penalised (strictly greater).
    fan_in_m: float = 0.85          # the multiplier applied above the threshold.
    file_rank_fn_anchor_boost: float = 1.0  # a file holding an exact-symbol
                                    # FUNCTION anchor keeps the
                                    # deterministic-beats-retrieval precedence
                                    # it had when anchors were prepended. NOT
                                    # earned on this bench (flat 0.0-5.0, the
                                    # gym has no task where an fn-anchor file
                                    # carries no retrieval evidence); it is here
                                    # so such a file cannot score 0.0 and sink
                                    # to last. Documented as untested, not
                                    # measured-optimal.


# Shapes for which block insertion/replacement is an intentional expression of
# the repair. Single-line attribute swaps, imports (banned by block_holes), and
# unknown classifications stay on their dedicated or fail-closed routes.
BLOCK_HOLE_SHAPE_ALLOWLIST = (
    "multi_line_coordinated",
    "guard_insertion",
    "exception_handler_change",
    "statement_insert",
    "literal_change",
)


@dataclass
class FixCfg:
    mutation_menu: bool = True
    # enumerators active. The four measured families + attr stay FIRST (the union
    # path enumerates in this order and truncates at menu_cap); the five contact
    # families append, closing the measured gold-contact-12 vs fixed-4 gap.
    families: tuple = ("cmp", "bool", "none", "arith", "attr",
                       "wrong_variable_used", "missing_kwarg", "missing_await",
                       "boolean_operator", "early_return_missing",
                       # mutations.py's attribute family, wired 2026-08-29. It
                       # swaps one attribute load for another the FILE already
                       # uses (fixer passes `file_lines`), which is the only way
                       # the real-history `.message` -> `.content` gold is
                       # reachable: menu size 8 with the file-wide pool, 60
                       # without. Distinct from `attr` above, which swaps
                       # against a task-named class's own source.
                       "wrong_attribute",
                       # EXPRESSIVE FAMILIES, wired 2026-08-29 off the real-19
                       # autopsy. Measured there: 1 of 19 real gold fixes was
                       # expressible by ANY enabled family, so the board's
                       # ceiling was 1-2 fixed at oracle-perfect locate. These
                       # three cover the shapes the autopsy's own diffs name —
                       # a key/constant rename, an `if x is None: continue`
                       # passthrough, and one clause added to a condition or a
                       # filter chain. They enumerate LAST, so a cap can only
                       # evict them.
                       "literal_substitution", "none_passthrough_guard",
                       "clause_insertion",
                       # COORDINATED FAMILY, wired 2026-08-29 (w27 multi-line
                       # lane, edit_sets.py). 10 of the 21 tasks on the v3 real
                       # board are `multi_line_coordinated`, a shape NO
                       # single-line family can express: the candidate is an
                       # ordered SET of edits applied atomically. Measured
                       # reach today is 3 of those 10 (three of the five
                       # committed coordination patterns —
                       # probes/ml_reachability.py prints the table and the
                       # named punts). Enumerated out of its OWN reserved
                       # budget after the per-function cap is already spent, so
                       # it cannot evict or reorder any earlier candidate.
                       "coordinated_edit_set")
    # ---------------------------------------------------------- TYPED HOLES
    # THE NOVEL-TOKEN BOUNDARY (holes.py, wired 2026-08-29). MEASURED on the v3
    # real board: 16 of 21 golds contain at least one token that occurs NOWHERE
    # in the parent file — invented locals (`_cc`, `_claimed_author`,
    # `_SAFETY_RELEVANT_FIELDS`), authored prose ('Turn dispatch failed'),
    # corpus-unique class names. `edit_sets.PUNTED_PATTERNS` and `CENSUS_PUNTS`
    # already classify every one of those misses NOVEL-TOKEN, and that
    # classification is right: an enumerator over the parent file can only ever
    # re-propose tokens the parent file already contains. More search does not
    # move it.
    #
    # So enumeration produces the STRUCTURE deterministically and leaves a typed
    # hole; a separate short no-think model call fills only that slot; the fill
    # is validated deterministically and INSTANTIATED into an ordinary EditSet
    # that enters the normal pipeline. Fills never bypass arbitration.
    #
    # SHIPS OFF ON MEASURED EVIDENCE — the same disposition, and the same
    # reason, as `LocateCfg.dunder_expansion`: BUILT, MEASURED and REFUSED as a
    # default, not skipped. The intent was ON by default under the fail-open
    # evidence rule. The evidence came back and said otherwise:
    #
    #   ORACLE-FILL REACHABILITY (probes/hole_reachability.py, 2026-08-29 —
    #   gold token injected, NO model, so this is the protocol's CEILING):
    #     17 of 21 v3 rows carry a novel token
    #      2 of those 17 differ from the parent ONLY in hole-shaped leaves
    #      1 of the 2 is reached  (row 8, `_env_guidance`, ORACLE@3)
    #     15 of the 17 are blocked by STRUCTURE the enumerator does not build
    #      6 exceed the structural hole ceiling
    #
    #   So the novel-token wall is real and this removes it, but it is NOT the
    #   wall these rows are standing behind. A perfect model filling a perfect
    #   hole still misses 16 of 17.
    #
    # Against a ceiling of one row, ON by default costs a fill call per hole per
    # template on EVERY task, and it perturbs 58 existing offline fixer tests
    # that mock `_call` (they would consume this stage's completions). Spending
    # fleet-wide budget and moving a measured harness to buy one row is not what
    # the fail-open rule is for — the rule exists so a mechanism with evidence
    # is not left dark, and here the evidence points the other way.
    #
    # TURN IT ON for the typed-hole experiment, and for any board whose golds
    # are leaf substitutions. Two consequences, both documented, neither silent:
    #   1. The tranche is APPENDED after `muts` is final, so it cannot reorder
    #      or evict an enumerated candidate at that point — but it DOES add
    #      candidates that `evidence_ranked` then sorts, so a tail entry can be
    #      pushed past `menu_cap`. Identical to what wiring any new family does.
    #      Knob OFF is byte-identical to the pre-holes harness (pinned by
    #      test_typed_holes_offline.py).
    #   2. `exclude_prose_candidates` drops `hole_literal_value` candidates,
    #      because they change text inside a string literal. That gate is older
    #      and better-measured than this family and is NOT exempted; the drop is
    #      counted into the certificate instead.
    typed_holes: bool = False
    typed_hole_templates: int = 12  # templates emitted per function
    typed_hole_samples: int = 3     # samples per hole; deduped, never retried
                                    # into existence (that would make pass@N
                                    # dishonest and the spend unbounded)
    typed_hole_fill_calls: int = 64  # HARD task-wide ceiling on model calls.
                                    # Generous per the step-generosity doctrine
                                    # and REPORTED in the cert either way, so an
                                    # exhausted budget is never mistaken for
                                    # "the model had nothing to say".
    typed_hole_grammar: bool = False  # GBNF on the fill call. OFF for the same
                                    # measured reason `KernelCfg.grammar_pick`
                                    # is off: grammar is incompatible with
                                    # thinking mode on this build and every
                                    # constrained call returned EMPTY. The
                                    # design's no-think fill call would fix it;
                                    # this stays off until ONE live call proves
                                    # it, and `holes.validate_fill` is the
                                    # authority regardless.
    # ----------------------------------------------------------- BLOCK HOLES
    # THE MULTI-STATEMENT BOUNDARY (block_holes.py, wired 2026-08-29).
    #
    # MEASURED on the v3 real board: of the 21 golds only 7 are single-block
    # ADDITIONS. Fourteen add statements AND rewrite or delete neighbouring
    # ones — a shape no single-line family expresses and, per
    # `blocks.BLOCK_PUNTS`, no block TEMPLATE expresses either (one template is
    # one contiguous inserted run). A contiguous-run REPLACEMENT is the same
    # single atomic splice as an insertion on the `edit_sets` substrate, so this
    # family enumerates BOTH, and that is the whole reason it is a distinct
    # family rather than more search inside an existing one.
    #
    # SHIPS ON for the allowlisted expressible shapes, under the fail-open
    # doctrine — the OPPOSITE disposition to `typed_holes` above, on different
    # evidence, not on a different mood:
    #
    #   ORACLE-BLOCK REACHABILITY (probes/block_hole_reachability.py — the gold
    #   block injected AS IF AUTHORED, no model, so this is the PROTOCOL
    #   CEILING and says nothing about live fill quality):
    #       10 of 21 v3 rows are reachable
    #        4 are out of scope by construction (they add a function-local
    #          import; `BLOCK_HOLE_PUNTS.banned_imports_are_permanent`)
    #
    #   Ten rows is not one row. `typed_holes` was refused as a default because
    #   a PERFECT model filling a PERFECT hole still missed 16 of 17; here a
    #   perfect author reaches half the board, and the whole gate pipeline —
    #   prose filter, inert filter, evidence rank, spec_exec, witness replay,
    #   evidence linkage — arbitrates every single fill with no bypass and no
    #   priority. A family that cannot skip a gate cannot ship a wrong fix past
    #   one, so the cost of being wrong here is a rejected candidate, and the
    #   cost of leaving it dark is ten rows.
    #
    # WHAT TURNING IT ON ACTUALLY CHANGES, stated rather than buried:
    #   1. Model calls. Up to `block_hole_sites` x `block_hole_samples` authoring
    #      calls per allowlisted shape, hard-capped by `block_hole_fill_calls`
    #      and REPORTED in the certificate either way, so an exhausted budget is
    #      never mistaken for "the model had nothing to say".
    #   2. Menu composition. The tranche is APPENDED after `muts` is final, so
    #      it cannot reorder or evict there — but it DOES add candidates that
    #      `evidence_ranked` then sorts, so a tail entry can be pushed past
    #      `menu_cap`. Identical to what wiring any new family does. Knob OFF is
    #      byte-identical to the pre-block harness (pinned by
    #      test_block_hole_wiring_offline.py).
    #   3. `block_holes.BLOCK_HOLE_ENABLED` stays False in that module and is
    #      NOT the authority: the fixer passes `enabled=True` explicitly. That
    #      module refuses to turn itself on by design, so exactly one place — a
    #      config knob a run's fingerprint records — decides.
    block_hole: bool = True
    block_hole_shapes: tuple = BLOCK_HOLE_SHAPE_ALLOWLIST
    # The tranche is shape-gated in fixer.py; this allowlist is deliberately
    # configurable so boards can opt into a different measured shape set.
    block_hole_sites: int = 8       # legal (position, shape) pairs offered per
                                    # function. `block_holes.MAX_SITES`.
    block_hole_samples: int = 3     # samples per site; deduped, never retried
                                    # into existence (that would make pass@N
                                    # dishonest and the spend unbounded)
    block_hole_fill_calls: int = 48  # HARD task-wide ceiling on authoring calls.
    block_hole_screen: int = 8      # w42 menu pressure: at most this many
                                    # DEDUPED candidates survive screening.
    block_hole_expose: int = 2      # ...and at most this many reach the menu.
                                    # Blocks are long options next to one-line
                                    # ones; exposing eight near-duplicates
                                    # would let one family crowd the ballot.
    exclude_prose_candidates: bool = True
                                    # Real-19 autopsy fix #4: a candidate whose
                                    # changed span lies inside a docstring,
                                    # string literal or comment can never be a
                                    # repair, yet 2 of the 3 wrong FIX_PROPOSED
                                    # rows on the real board were exactly that
                                    # and passed every gate. Judged against the
                                    # WHOLE FILE's token stream, not one line.
    menu_cap: int = 14
    evidence_ranked: bool = True    # task-symbol overlap ordering
    mentioned_attr_first: bool = True
    exclude_const_attrs: bool = True
    none_option: bool = True        # menu carries "none of these"
    witness_replay: bool = True     # code replays boolean candidates on boundary
                                    # witnesses; unique survivor -> no model call
    spec_exec: bool = True          # SPECULATIVE EXECUTION AS ARBITER
                                    # (spec_exec.py). The mutation candidate set
                                    # is CLOSED, so before the menu is rendered
                                    # every candidate is applied in isolation and
                                    # the deterministic battery is run: compile,
                                    # witness replay, ONE synthesized
                                    # discriminating test over the whole menu,
                                    # and the symptom's own named direction.
                                    # A unique full survivor carrying AFFIRMATIVE
                                    # evidence is DECIDED and taken with zero
                                    # model calls (the ballot, the staged call
                                    # and the menu are all gated on new_src);
                                    # several survivors SHORTLIST the menu; zero
                                    # survivors leave it untouched. Measured on
                                    # the bench sweep: code-decides 7 tasks
                                    # including 5 of the current chronics. OFF
                                    # restores the byte-identical prior path.
    line_ballot: bool = True        # rung 1b: one INDEPENDENT yes/no call per
                                    # distinct candidate line, used only to
                                    # REORDER the mutation menu. Measured
                                    # 2026-08-28 (cfg 543b771d3cd4 autopsy): the
                                    # mutation menu was not a decision at all —
                                    # the model answered option A in 11 of the 12
                                    # menus it read, and `fixed` correlated
                                    # perfectly with gold happening to sit at
                                    # index 0 (8/18). The ballot has no position
                                    # dimension and names the right line 13/18.
    line_ballot_tau: float = 0.05   # decisiveness gate: reorder only when the
                                    # winning line beats the incumbent by this
                                    # much signed-confidence. Swept 0.02-1.5 on
                                    # the measured set: 0.02-0.08 is a FLAT
                                    # plateau at 12/18 with 4 conversions and 0
                                    # regressions (0.1-0.3 -> 11/18, 0.4+ -> 10),
                                    # so 0.05 is the plateau midpoint rather than
                                    # one lucky point. Its job is to veto the
                                    # SATURATED ballots: the 1B's yes-bias ties
                                    # lines at p=+-1.0, and the live cycle's
                                    # saturated gaps measured 0.0 and 0.0001 —
                                    # ~500x below this tau, so they stay vetoed —
                                    # while api/transcripts.py favoured the GOLD
    line_ballot_context_lines: int = 11  # numbered function context per ballot
    line_ballot_context_chars: int = 900  # visible cap, marked when clipped
                                    # line by 0.066 and was wrongly blocked at
                                    # the initial 0.30.
    line_ballot_max_calls: int = 8  # hard per-task ceiling on ballot calls.
                                    # RAISED 4 -> 8 on the 2026-08-29 ladder
                                    # autopsy (/tmp/trace_holistic.md fix #1):
                                    # menus with 5+ distinct candidate lines
                                    # were being decided on a PARTIAL ballot,
                                    # i.e. the winner was chosen from a subset
                                    # that need not contain the gold line.
                                    # menu_cap is 14, so 8 covers every distinct
                                    # -line count the gym produces while still
                                    # being a hard ceiling.
    line_ballot_select: bool = True  # BALLOT SELECTS, not merely re-ranks
                                    # (autopsy fix #1, the top conversion).
                                    # Measured over the 21-task ladder (84
                                    # runs): the ballot's argmax line is the
                                    # gold line 22/45 (49%) while the mutation
                                    # menu picks it 28/65 (43%) — and on the
                                    # four CHRONIC sibling-line tasks
                                    # (log_handler::_map_level,
                                    # image_processing::validate_file_count,
                                    # math_answer_verifier::_build_equivalent_forms,
                                    # curriculum_framework::_grade_to_band) the
                                    # ballot is right at EVERY budget while the
                                    # menu is wrong at every budget. Re-ranking
                                    # handed control straight back to the worse
                                    # decider. With this on, a decisive ballot
                                    # FILTERS `muts` to its winning line and the
                                    # menu only chooses among that line's
                                    # variants. Nothing else changes: the menu,
                                    # its margin probe, the reversed retry and
                                    # every gate still run. OFF restores the
                                    # stable-partition reorder exactly.
    line_ballot_select_tau: float = 0.15  # SELECTION IS GATED HARDER THAN
                                    # REORDERING, and this is the whole safety
                                    # story. Measured on the armed cycle
                                    # (2026-08-29): the ballot is WRONG on 2 of
                                    # the 4 chronic tasks —
                                    # image_processing prefers line 307 (0.79)
                                    # over gold 312 (0.71), a gap of 0.08, and
                                    # curriculum_framework saturates at ~0.9996
                                    # on THREE lines while gold is a fourth.
                                    # Filtering `muts` to the argmax there would
                                    # hard-lock the wrong line and convert a
                                    # 43%-correct menu into a 0%-correct filter.
                                    # So selection needs a REAL margin (this
                                    # tau, 3x the reorder tau) on an UNSATURATED
                                    # argmax; anything weaker degrades to the
                                    # reorder-only incumbent, which keeps both
                                    # lines on the menu. That is the
                                    # "disagreement escape hatch": when the
                                    # ballot and the evidence-ranked incumbent
                                    # disagree without conviction, neither side
                                    # wins outright.
    line_ballot_select_factor: float = 2.0  # SELECTION PRUNES, SO IT CLEARS A
                                    # DOUBLE BAR (rung-4000 autopsy, ranked fix
                                    # #3). `decision: "selected"` deletes every
                                    # other line from the menu, and it deleted
                                    # the GOLD line twice on the 12-failure
                                    # board: book_series dropped 1 (leaving a
                                    # single option on line 102) and
                                    # teacher_personas dropped 3 (leaving one
                                    # plausible decoy the model then argued for
                                    # correctly). A prune cannot be recovered
                                    # from downstream, so the margin must EXCEED
                                    # factor x select_tau, and a uniquely
                                    # saturated winner no longer bypasses that
                                    # bar. Set to 1.0 to restore the previous
                                    # single-tau behaviour.
    line_ballot_saturation: float = 0.999  # p(yes) at or above this is a
                                    # SATURATED vote. The 1B's yes-bias pins
                                    # several lines at ~1.0 simultaneously and
                                    # the gaps between them collapse to ~1e-4,
                                    # which carries no information at all.
    line_ballot_saturated_decisive: bool = True  # a ballot saturated at
                                    # p(yes)=1.0 on EXACTLY ONE line is a
                                    # decision, not a tie (autopsy fix #1, third
                                    # bullet). The tau gate exists to veto
                                    # MULTI-line saturation, where the 1B's
                                    # yes-bias ties several lines at +-1.0 and
                                    # the gap collapses to ~0. A unique
                                    # saturated line has no such ambiguity.
    line_ballot_flat_epsilon: float = 0.001  # a ballot whose scores all fall
                                    # within this SPREAD carries no signal at
                                    # any level. MEASURED: `log_handler`'s four
                                    # probed lines scored -0.9998/-0.9997 three
                                    # cycles running — flat, but nowhere near
                                    # the +0.999 absolute saturation test, so it
                                    # was ordered by float noise and its (gold)
                                    # winner then discarded as "not decisive".
    # ---- MUTATION-MENU SELECTOR (conversion autopsy, 2026-08-29, §3) --------
    # MEASURED on the half-A real bench (`/tmp/conversion_autopsy.md`): the
    # A-H letter menu is a POSITION READOUT, not a decision. 24 of 25 mutation
    # menus resolved to option A, unanimously 3/3, at logprob margins of
    # 0.674..8.167 — every one an order of magnitude above `margin_tau=0.15`, so
    # the reversed-menu position-bias guard was structurally unreachable and
    # fired ZERO times. Under an A-lock only a task whose gold sits at survivor
    # index 0 can convert; across 11 tasks gold was at index 0 zero times, and
    # `fixed=0` was therefore decided before any model reasoning mattered.
    #
    # "binary" replaces the letter menu as the SELECTOR with one independent
    # yes/no probe per candidate over the rendered `old -> new` edit. It is the
    # only framing in that run where the model produced varied output at all
    # (letter menus: 24/25 identical; binary probes: -0.995..+1.0). Binary
    # probes have no position dimension by construction.
    #
    # E49 DOCTRINE PRESERVED: code ranks (by probe score, with the incoming
    # evidence order as the only tie-break), the model merely CONFIRMS or VETOES
    # the top. "letter" restores the previous path byte-identically.
    menu_selector: str = "binary"   # "binary" | "letter"
    # THE SAME SELECTOR AT THE TWO REMAINING LETTER-MENU STAGES (E100 port).
    # `menu_selector` above governs the MUTATION menu it was built for; these two
    # govern the FILE and FUNCTION menus, which measurably A-locked on their own
    # (E100: the file stage picked index 0 twice with gold at index 3, and every
    # results-ledger row since 1788010000 carries the shape).
    #
    # They are separate knobs on purpose. The three stages differ in what a probe
    # is shown (an edit / a file's matching lines / a function body) and in what a
    # probe costs, so a calibration that holds at one is not evidence about the
    # others, and a lane that wants to isolate one stage must be able to.
    # "letter" restores each stage's previous path byte-identically.
    #
    # EVERY OTHER binary_probe_* knob below is SHARED by all three stages —
    # deliberately, including `binary_probe_confirm_ratio`, which is NOT retuned
    # here: it was calibrated on mutation-menu rows and is due a re-derivation
    # after a full clean cycle. Applying the same mechanism with the same number
    # is the honest move; inventing a per-stage number from no data would not be.
    # FILE STAGE: SHIPPED BACK ON "letter" 2026-08-29 ON MEASUREMENT.
    # The binary port was armed, run on 7 real gym tasks, and produced 7/7
    # ABSTAIN with contact 0 (walls 78-104s) where the letter menu had contact
    # 3/6 on the same tasks. TWO defects, both visible in
    # `stages[].file_selector` on the rows since epoch 1788022000:
    #
    #   1. THE CONFIRM VETOED 8/8 (see FILE_CONFIRM_Q's note) — an unfalsifiable
    #      question with veto power. FIXED in the wording, and the location
    #      confirm now ships OFF (`location_probe_confirm`).
    #   2. THE PROBE FIELD SATURATES, which the fix for (1) does NOT cure. The
    #      file question is a plausibility question, and at 1B almost every
    #      candidate earns a yes: 6-8 of 8 candidates scored >= 0.85, most of
    #      them >= 0.99. On `ea639fc1` the GOLD file scored 0.9765 and LOST to
    #      `utils/grading_service.py` at 0.9960 — the confabulation attractor
    #      named in the project rules, reproduced exactly. A 0.0195 gap between
    #      the right answer and a known attractor is not a ranking, it is noise
    #      with a decimal point.
    #
    # WHY THE MUTATION STAGE IS DIFFERENT, and why this is not an argument
    # against the selector as such: a mutation probe compares two concrete
    # texts (before -> after) and asks whether a MECHANISM changed — visible or
    # not, per candidate. A file probe asks whether a defect COULD live in a
    # file, which is a judgement about plausibility that a 1B answers from term
    # mass. Position-freedom does not help when the scores it produces are
    # undifferentiated.
    #
    # "binary" stays fully wired and tested so a calibration lane can attack the
    # probe WORDING (a disqualifying rule with teeth, as CAND_PROBE_Q has) and
    # re-measure. Flip it back only with a contact number that beats letter.
    # Pairwise selection uses a champion
    # tournament of A/B/TIE COMPARISONS over the code ranker's order, N-1 calls
    # plus one position-swapped rematch when the model displaces the ranker's
    # leader. It exists because the failure measured above is not a bug in the
    # plumbing — an absolute "could the bug be here" question cannot discriminate
    # at 1B no matter how it is worded, while a comparison between two concrete
    # cards is the same shape as the mutation probe that does work.
    # BUILT, TESTED, AND DORMANT. It has never been run against the live model,
    # so it ships off; promoting it needs the criterion — more runner-verified
    # fixes than the ranker alone, not a better pairwise accuracy number.
    file_selector: str = "letter"       # "letter" | "binary" | "pairwise"
    # FUNCTION STAGE: ALSO SHIPPED ON "letter", 2026-08-29, on its OWN measurement
    # rather than by analogy. When the file stage was put back on letter the run
    # finally reached the function stage, and the first live `function_selector`
    # record showed the SAME pathology as the file probes:
    #   ea639fc1, 4 candidates, affirmed_frac 1.0 (every candidate affirmed),
    #   gold `_build_chat_block` 0.6616 -> LOST to `_embed_memory_best_effort`
    #   0.9964, a fire-and-forget embedding helper with nothing to do with the
    #   reported attribute swap.
    # The `margin` of 0.3347 looks healthy, which is exactly why `affirmed_frac`
    # now travels with it: a confident-looking win over a field the model
    # affirmed wholesale is not a decision, it is the same plausibility read.
    #
    # WHAT THE LETTER PATH HAS THAT THIS DOES NOT: `function_rank_by_evidence`
    # already puts gold at option A in 19/21 measured cases. Replacing a ranker
    # that is usually right at position 0 with a probe that ranks gold third of
    # four is strictly worse, A-lock or no A-lock.
    #
    # EVIDENCE HONESTY: this is ONE task, not a bench. It is flipped because the
    # signature matches the eight file rows exactly and because the cost of being
    # wrong is another empty bench; a calibration lane with a sharper probe
    # question should re-measure both stages together.
    function_selector: str = "letter"   # "binary" | "letter"
    location_probe_confirm: bool = False  # the confirm/veto call at the FILE and
                                     # FUNCTION stages. OFF on measurement: across
                                     # the 8 live rows it vetoed 8/8 and confirmed
                                    # 0/8, i.e. it only ever destroyed a
                                    # selection. The wording that caused that is
                                    # fixed above, but a veto is a mechanism with
                                    # a one-way cost — it can only turn a pick
                                    # into an abstain — so it stays off until a
                                    # lane measures the NEW question and shows it
                                    # discriminating. The MUTATION confirm
                                    # (`binary_probe_confirm`) is untouched and
                                     # still ON: it is measured, it declines weak
                                     # vetoes, and it confirms real winners.
    probe_scoring: str = "logit_margin"  # "token" keeps the sampled-token
                                     # score; "logit_margin" uses
                                     # logP(YES)-logP(NO) from the answer
                                     # position's top_logprobs. The latter is
                                     # the default because sampled-token sign
                                     # errors inverted the measured gold/decoy
                                     # ranking.
    binary_probe_max: int = 12      # hard ceiling on per-candidate probes. A
                                    # candidate past the cap is never probed and
                                    # so can never win; the cert records it.
    binary_probe_confirm: bool = True   # one final yes/no on the ranked winner.
                                    # A NEGATIVE confirm VETOES (honest abstain);
                                    # an unreadable one fails OPEN and is recorded.
    binary_probe_margin_scale: float = 4.0  # nats. Probe scores are the
                                     # class-mass YES-vs-NO logit margin
                                     # squashed by `tanh(m / scale)` — see
                                     # `_yes_no_margin`.
                                    # The squash is monotone, so this changes no
                                    # ordering; it only sets how quickly a
                                    # confident probe saturates toward +-1, which
                                    # is what keeps one shouted answer from
                                    # dominating the field. Raise it to spread a
                                    # saturated field, lower it to sharpen a
                                    # timid one. Deliberately NOT applied to the
                                    # line ballot, whose taus were measured
                                    # against the older signed-probability scale.
    binary_probe_confirm_ratio: float = 0.25  # HOW MUCH the confirm probe must
                                    # carry before it may overturn the ranked
                                    # winner. The veto fires only when
                                    # `-confirm_score >= ratio * winner_score`,
                                    # i.e. a veto must be worth at least a
                                    # QUARTER of the evidence the selection was
                                    # worth. `0.0` restores the pre-2026-08-29
                                    # bare-sign veto exactly (`-cs >= 0` is
                                    # always true), so this is a strict
                                    # generalisation and the old behaviour stays
                                    # reachable.
                                    #
                                    # CALIBRATED, not chosen. Every
                                    # `mutation_selector` record in
                                    # results/unified_gym.jsonl that ran a
                                    # confirm probe (|cs| / winner-score):
                                    #   0.008  2dbb4371   0.052  fb1d75cb
                                    #   0.098  97062f14   | 0.528 ea639fc1
                                    #   0.942  c63265b3   0.994 c8a6ca03(1.039)
                                    #   1.142  a9af5404   1.156 7787d96f
                                    #   1.260  09523282
                                    # The field is BIMODAL with an empty band
                                    # between 0.098 and 0.528; 0.25 sits in that
                                    # band with >2.5x headroom on both sides. The
                                    # three rows below the band were vetoed by a
                                    # confirm score of -0.0068 / -0.0416 /
                                    # -0.0958 — noise overturning winners scored
                                    # 0.875 / 0.808 / 0.977, one of them with a
                                    # 0.546 margin over its runner-up. The six
                                    # above the band keep vetoing unchanged.
                                    # A confirmed winner (0a492660, cs=+0.1256)
                                    # is a POSITIVE score and never reaches this
                                    # predicate at all.
    binary_probe_flat_epsilon: float = 0.001  # a probe field whose scores all
                                    # fall within this SPREAD says the same thing
                                    # about every candidate, i.e. nothing. Same
                                    # reasoning as `line_ballot_flat_epsilon`.
    line_fix_fallback: bool = True  # rung 2: region + single-line emission
    # LINE_STAGE (architecture/line-stage-seam.md, wired 2026-08-29). "auto"
    # routes on the EVIDENCED candidate count: enumerate when the function is
    # small enough to enumerate for free, menu when it is not. p_max/m_max are
    # the seam's own pre-measurement over 21 real v3 tasks: gold-in-P is 21/21
    # at p_max=120 and 19/21 at 60/80, and gold-in-menu flattens at m_max=12.
    # The seam doc's WIRING constant says 60 for p_max; §4 of the same doc
    # measured 120 and that is the number shipped — the constant predates the
    # sweep. `off` restores the pre-wiring single-line path byte-for-byte.
    line_stage_path: str = "auto"   # "off" | "auto" | "menu" | "enumerate"
    line_stage_p_max: int = 120
    line_stage_m_max: int = 120     # 12 -> 120 by MEASUREMENT (E146, 2026-08-30,
                                    # /tmp/RETRIEVAL_COVERAGE.md). The earlier 12
                                    # came from "gold-in-menu flattens at m_max=12"
                                    # — true for the MENU, but it interacted with
                                    # skip_above=16 to render NO model-facing
                                    # anchor window at all on 8 of 21 tasks. Gold
                                    # was in the file menu 21/21 and the function
                                    # menu 21/21, then died at the line window.
                                    # Raising both took final-window containment
                                    # from 12/21 to 21/21, converting 9 tasks.
                                    # The model was abstaining on bugs it was
                                    # never shown.
    line_stage_skip_above: int = 120  # was 16. Same measurement: at 16 the stage
                                    # judged a function too large to display and
                                    # showed the model nothing, silently.
    # GET_SURE (detreason/get_sure.py's seam, wired 2026-08-29 in
    # probes/unified_gym.py). After an ELIGIBLE abstain, deterministic evidence
    # mined from the run's own certificate is appended to the task and the
    # WHOLE decision stack is re-run through `fixer.run` -- never a partial
    # reuse, which is the one way the loop could bypass or double-run
    # line_stage.
    #
    # SHIPPED OFF, and the reason is measured, not cautious: one round is one
    # extra FULL decision, and the live smoke on this board measured 266-326s
    # per decision. On the 21-task board roughly half the tasks abstain, so a
    # single round costs about +45-60 minutes of wall on a board that is
    # already the critical path. Bench E (get-sure only) is where that is
    # bought deliberately; flipping it on here would silently fold the cost
    # into every other arm's number.
    get_sure: bool = False
    get_sure_rounds: int = 1
    retries: int = 4
    framings: int = 1               # >1 rotates task framings across retries
    microcall_line_link: bool = True  # ARMED 2026-08-29. detreason/microcall.py
                                    # site #1 (skeptic.symptom_line_link), wired
                                    # with a real transport over fixer._post.
                                    # Fires ONLY when the mutation menu's own
                                    # top-2 margin came back thinner than
                                    # kernel.margin_tau — i.e. exactly the
                                    # fragile menus the site exists to help. The
                                    # site's VERIFIED answer appends a
                                    # " | symptom-link" suffix to the matching
                                    # option and the menu is re-asked; an
                                    # unverified answer (the layer's fallback
                                    # path) changes nothing at all. Append-only:
                                    # option membership and order are untouched.
    staged_reasoning: bool = True   # Enable staged reasoning (
                                    # first live test). ONE reasoning-only call
                                    # before the mutation menu asking which
                                    # candidate line most likely holds the bug
                                    # and why. Its output is SANITIZED (braces
                                    # stripped per the standing rule, capped at
                                    # staged_reasoning_max_chars) and appended to
                                    # the mutation-menu prompt as
                                    # "PRIOR ANALYSIS: ...". It never selects a
                                    # repair, never reorders the menu and never
                                    # bypasses a gate — it is prompt context
                                    # only. Fail-open: any error leaves the
                                    # prompt byte-identical.
    staged_reasoning_max_chars: int = 300
    info_requests: bool = True      # BOUNDED INFO-REQUEST AT DECISION POINTS
                                    # (design a_info_request). E60-E66 traces
                                    # show the 1B reasoning "let me check the
                                    # actual code in A" mid-menu with no way to
                                    # ask, so it guesses or abstains. With this
                                    # on, a menu answer may be ONE of pick /
                                    # request / abstain; a request is validated
                                    # against the CURRENT menu (enum + target,
                                    # never free-form), fulfilled from code the
                                    # harness already has (func_index spans,
                                    # docstrings, AST call scan, span slicing)
                                    # and appended to a re-ask as ADDITIONAL
                                    # EVIDENCE.
                                    # SHIPS ON, deliberately, against the E60
                                    # parking-brake doctrine: this feature is
                                    # fail-open and EVIDENCE-ONLY — it cannot
                                    # select, reorder, drop or add a candidate,
                                    # and cannot bypass a gate. The worst case
                                    # is `info_request_budget_per_menu + 1`
                                    # decision rounds. Knob OFF is
                                    # byte-identical to the pre-request harness
                                    # (pinned by a test).
    info_request_budget_per_menu: int = 1   # requests admitted per menu before
                                    # `requests_closed` drops REQUEST from the
                                    # affordance and one FINAL pick-or-abstain
                                    # round is asked. The attention allocator's
                                    # per_stage[MENU] may RAISE this for an
                                    # uncertain task; nothing can lower it.
    info_request_budget_per_task: int = 2   # hard task-wide ceiling. Charged on
                                    # ADMISSION — malformed, duplicate and
                                    # unfulfillable requests all cost one, and
                                    # there is no refund, because a refund is a
                                    # loop with extra steps.
    info_request_context_lines: int = 12    # per-fetch line cap
    info_request_evidence_max_chars: int = 1200  # whole-block prompt cap


@dataclass
class GateCfg:
    compile_gate: bool = True
    polarity_probes: bool = True
    regression_asserts: bool = True
    evidence_linkage: bool = True   # unlinked change -> LOW_CONFIDENCE_PROPOSAL
    # `symptom_identifier_lexical_v1` (w38 `a_evlink_s.md`), wired 2026-08-29.
    # An IDF-weighted overlap between the symptom's nouns and the identifiers
    # the edit ADDED. ADDITIVE ONLY: it is the LAST disjunct of the linkage
    # gate, evaluated solely when every stronger channel has already declined,
    # so it can neither replace a channel nor change what one decides.
    #
    # SHIPS ON. It is strictly a widening of an existing gate's ACCEPT set, and
    # the thing that keeps that safe is not the knob but the construction: the
    # IDF weighting means a hub file's common identifiers score near zero, and
    # BOTH a rarity floor and a 60%-coverage floor must hold. Measured against
    # the case that matters — the real-19 attractor edit — it still REFUSES
    # (pinned by test_evidence_idf_channel_offline.py).
    #
    # STRENGTH 1 IS THE CEILING: this clears "no deterministic link", never
    # behavioural verification, and may not on its own satisfy any promotion or
    # behavioural gate. Knob OFF is byte-identical to the pre-channel harness.
    evidence_idf_channel: bool = True
    reemit_guard: bool = True       # AST-diff on full-function re-emission
    patch_relevance: bool = True    # class-relevant AST delta + no-op guard


@dataclass
class KernelCfg:
    """Tri-valued atom kernel (B1). Sensors answer; code verdicts."""
    enabled: bool = True
    unknown_threshold: float = 0.30  # calibrated margin below which -> UNKNOWN
    margin_probe: bool = True       # n_probs top-2 logprob fragility detection
    margin_tau: float = 0.15         # nats; calibrate per 10-boot recipe
    paired_entailment: bool = False  # P(claim) + P(negation) probes
    variant_verify: int = 0          # extra order/polarity-flipped probes per atom
    hats: int = 0                    # 0=off; N = family-routed hat ensemble size
    win_margin: float = 0.10         # interval-scoring winner margin
    menu_votes: int = 3              # MAJORITY VOTING over within-boot jitter.
                                     # MEASURED: 7 of 21 gym tasks flip pass/fail
                                     # across IDENTICAL runs — same boot, same
                                     # seed, same prompt. The server's continuous
                                     # batching reorders float reductions, so a
                                     # near-tie menu lands on either side. That
                                     # jitter is not noise to suppress: it is a
                                     # FREE sampler over the model's own
                                     # uncertainty. N>1 asks the SAME menu N times
                                     # (identical prompt — batching supplies the
                                     # variance) and takes the majority; the tally
                                     # itself measures how close the call was.
                                     # No majority -> the existing thin-margin
                                     # reversed-retry/abstain path, which is the
                                     # right answer for a genuine coin flip.
                                     # 1 = pre-voting behavior, byte-identical
                                     # (pinned by a test). Tokens are free here,
                                     # so 3x the menu calls is sanctioned.
    retry_tiebreak: bool = True      # THE REVERSED RETRY IS A TIE-BREAK, NOT A
                                     # VETO (rung-4000 autopsy, ranked fix #1).
                                     # MEASURED on the 12 failures: the probe
                                     # fired 6 times, rescued 0, and converted 4
                                     # would-be gold contacts into abstains; in 5
                                     # of 6 firings the correct answer sat on one
                                     # side of the retry and `disagree -> abstain`
                                     # discarded it. With this ON, disagreement
                                     # resolves by weight of evidence (see
                                     # Fixer._arbitrate): an option named on BOTH
                                     # sides wins, a disjoint retry loses to a
                                     # pre-retry majority, and only a no-majority
                                     # vote that the retry also contradicts still
                                     # abstains. OFF restores the legacy veto, so
                                     # the A/B is exactly one knob wide.
    retry_skip_on_unanimous: bool = True  # do not CALL the reversed retry when
                                     # the pre-retry vote was unanimous and the
                                     # pick survived the witness gate. MEASURED:
                                     # image_processing (3/3 on the gold file) and
                                     # code_submissions (3/3 on the byte-exact
                                     # gold mutation) were both vetoed by a
                                     # position swap. A unanimous menu gives the
                                     # reversal nothing to measure, so the call
                                     # can only veto — and it costs a call to do
                                     # it. Witness admission is the guard that
                                     # keeps a unanimous `none of these` (which is
                                     # never witness-admitted) still challenged.
    grammar_pick: bool = False       # MEASURED LIVE 2026-08-30: GBNF grammar is
                                     # INCOMPATIBLE with thinking mode — grammar
                                     # forbids the reasoning block's opening, the
                                     # model can emit nothing legal, every call
                                     # returns EMPTY with finish_reason=stop
                                     # (21/21 instant abstains). Lenient parser +
                                     # one-letter re-ask + #11d headroom cover
                                     # the C5 class instead. Re-enable only with
                                     # a grammar that permits the think block,
                                     # proven by ONE live call first.
                                     # ORIGINAL RATIONALE (autopsy fix #2,
                                     # /tmp/trace_holistic.md). 27 of 84 ladder
                                     # runs — a third of everything — died with
                                     # pick=None because the 1B answered in
                                     # PROSE or spilled its reasoning into
                                     # `content`; in several the prose contained
                                     # the CORRECT answer. So constrain the
                                     # channel instead of parsing harder: every
                                     # menu call carries a GBNF grammar whose
                                     # only legal completion is
                                     # {"pick": "<letter>"}. llama.cpp accepts
                                     # `grammar` in the chat-completions body
                                     # and ignores unknown keys, and a server
                                     # that REJECTS it is retried once with the
                                     # key removed, so this is fail-open by
                                     # construction and never costs a call.
    grammar_reask: bool = True       # the fallback half of fix #2 for servers
                                     # with no grammar support: on pick=None
                                     # with NON-EMPTY prose, re-ask once with
                                     # the model's own prose prepended and a
                                     # tiny budget, demanding one bare letter.
                                     # Never fires on an empty answer (that is a
                                     # transport failure, not a formatting one)
                                     # and never invents a letter the menu did
                                     # not offer.
    capture_traces: bool = True      # persist the model's THINKING trace for every
                                     # completion to results/traces/<key>.jsonl.
                                     # Traces are big, so the certificate carries
                                     # only a {file,line} ref + a token count.
                                     # Fail-soft: capture never affects a call.


@dataclass
class FactsCfg:
    inject_class_attrs: bool = True
    inject_task_named_files: bool = True
    comments_as_facts: bool = False  # typed comment channel w/ precomputed age
    comment_stale_months: int = 6    # blame-age gap that flags a comment stale
    brace_sanitize: bool = True      # standing rule: brace literals hang decisions
    intent_oracle: bool = True       # Mine executable expectations
                                     # from the function's own docstring +
                                     # signature (no model call) and use them to
                                     # kill mutation candidates that contradict
                                     # the stated contract. Attacks the E50
                                     # bottleneck directly: witness replay cannot
                                     # separate cmp-flip siblings because nothing
                                     # in the sandbox encodes INTENT; the prose
                                     # does. Ships OFF — promotion belongs to the
                                     # cycle_report causal gate, not to a builder's
                                     # optimism. Knob OFF is byte-identical to the
                                     # pre-oracle path (pinned by a test).


@dataclass
class AttentionCfg:
    """Attention/depth allocator (detreason/attention.py, design a_attention).

    The allocator is built,
    tested and deterministic; fixer.py now READS it at task start and maps
    `per_stage` onto this run's ballot budget and menu-retry values. Knob OFF is
    still byte-identical to the pre-allocator path (pinned by a test).

    The floors are the reason this feature exists: uncertainty buys depth, and a
    confidently-WRONG upstream pick asks for nothing, so a purely
    uncertainty-driven allocator defunds the verification that would have caught
    it. Floors are allocated first and never scale with confidence. They are
    raiseable and NEVER lowerable from here.
    """
    enabled: bool = True
    stakes_tier: str = "medium"     # low|medium|high|critical; scales the stake
                                    # of the VERIFICATION stages only (a flat
                                    # multiplier cancels out of every argmax)
    budget: float = 24.0            # total price units per task
    reserve_fraction: float = 0.15  # post-floor budget held for preemption
    hysteresis_ratio: float = 1.25  # challenger must beat the standing plan by
    hysteresis_checkpoints: int = 2 # ...this many consecutive checkpoints
    floor_retrieval_width: int = 2      # independent retrieval paths
    floor_menu_retries: int = 1         # retries beyond the first attempt
    floor_ballot_budget: int = 1        # independent line votes
    floor_witness_inputs: int = 2       # independent witnesses — never zero
    floor_deliberation_branches: int = 2  # branches incl. the skeptic — never zero


@dataclass
class PromptLedgerCfg:
    """Durable prompt/response capture for every fixer model-call attempt."""

    enabled: bool = True
    directory: str = ""            # empty = results/prompt_ledger beside fixer.py
    max_bytes: int | None = None       # module default or PROMPT_LEDGER_MAX_BYTES


@dataclass
class LLMCfg:
    """Measured model-call budgets and thinking policy, by stage.

    `max_tokens` is a total completion cap on the reasoning server, so these
    answer sizes are deliberately separate from the hidden-reasoning headroom.
    """

    reasoning_headroom: int = 1536
    reasoning_multiplier: float = 2.0
    menu_answer_tokens: int = 96
    relevance_answer_tokens: int = 16
    binary_probe_answer_tokens: int = 16
    line_probe_answer_tokens: int = 16
    line_ballot_answer_tokens: int = 16
    staged_reasoning_answer_tokens: int = 128
    line_emission_answer_tokens: int = 400
    hole_answer_tokens: int = 32
    block_answer_tokens: int = 320
    microcall_answer_tokens: int = 32
    reask_answer_tokens: int = 8
    menu_thinking: bool = True
    relevance_thinking: bool = True
    binary_probe_thinking: bool = True
    line_probe_thinking: bool = True
    line_ballot_thinking: bool = True
    staged_reasoning_thinking: bool = True
    line_emission_thinking: bool = True
    authoring_thinking: bool = True
    microcall_thinking: bool = False
    reask_thinking: bool = False


@dataclass
class GymConfig:
    locate: LocateCfg = field(default_factory=LocateCfg)
    fix: FixCfg = field(default_factory=FixCfg)
    gates: GateCfg = field(default_factory=GateCfg)
    kernel: KernelCfg = field(default_factory=KernelCfg)
    facts: FactsCfg = field(default_factory=FactsCfg)
    attention: AttentionCfg = field(default_factory=AttentionCfg)
    prompt_ledger: PromptLedgerCfg = field(default_factory=PromptLedgerCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    # MEASURED FOOTGUN (2026-08-30, FINDINGS E244): this default was 8098, where NOTHING
    # listens. 130 of 141 ERROR rows in a 1,417-row window were `<urlopen error [Errno 61]
    # Connection refused>` — ~9% of the whole bench spent on a dead port, scored as task
    # failures. Live endpoints are 8096 (local MiniCPM5-1B) and 8097 (port-forward to the
    # EKS GPU pod `gym-llama`, serving qwen4b.gguf). 8097 is the bench's model, so it is the
    # default. GYM_URL overrides for a local-only run.
    #
    # Env-overridable ON PURPOSE: the port-forward is a tunnel and tunnels die. A hardcoded
    # host:port is exactly the kind of silent-drop dependency that turned infrastructure
    # death into "the model failed" for an unknown number of prior runs.
    url: str = os.environ.get(
        "GYM_URL", "http://127.0.0.1:8097/v1/chat/completions")
    workers: int = 1
    seed: int = 31

    def fingerprint(self):
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_dict(self):
        return asdict(self)


def default() -> GymConfig:
    return GymConfig()
