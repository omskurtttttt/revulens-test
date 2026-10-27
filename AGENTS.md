# Project Directives: Installed Skills & Always-On Practices

Laging tandaan at gamitin ang mga sumusunod na core skill sets at practices sa bawat task sa repository na ito (`revulens-test`):

---

## 1. Andrej Karpathy Guidelines (`karpathy-guidelines`)
- **Think Before Coding**: Huwag manghula ng assumptions. Kung may confusion o tradeoffs, sabihin agad at magtanong. I-surface ang tradeoffs bago mag-implement.
- **Simplicity First**: Isulat lamang ang pinakasimpleng code na lumulutas sa problema. Walang speculative features, walang unrequested abstractions o premature configurability.
- **Surgical Changes**: Hawakan lamang ang kinakailangang baguhin. Huwag galawin o i-reformat ang unrelated code. Panatilihin ang existing style ng codebase. Linisin lamang ang mga sariling nilikhang orphans (unused imports/vars).
- **Goal-Driven Execution**: Magtakda ng malinaw at verifiable success criteria (halimbawa: step -> check) at i-verify bago ituring na tapos.

---

## 2. Ponytail - Lazy Senior Dev Mode (`ponytail`, `ponytail-review`, `ponytail-audit`)
- **Smallest Complete Change**: Ang pinakamagandang code ay ang code na hindi na kailangang isulat. Gamitin muna ang kung anong meron na: existing helper, standard library, o installed dependency bago magdagdag ng bago.
- **Strict Guardrails**: Hinding-hindi babawasan ang validation, security boundaries, error handling, at testing para sa risky logic.
- **Transparent Summary**: Tapusin ang update o reply sa pamamagitan ng pagsasabi kung ano ang hindi na-check o na-skip at ano ang mga posibleng risk na dapat malaman ng user.
- **Review & Audit**: Gamitin ang `/ponytail-review` at `/ponytail-audit` para matiyak na walang bloat, regressions, o dead ends.

---

## 3. Graphify (`graphify`)
- Gamitin ang Graphify para sa code comprehension, call graphs, concept explanations, at architectural mapping.
- Kapag kailangang intindihin ang ugnayan ng mga modules, gamitin ang graph analysis para hindi nanghuhula o nag-i-scan nang paulit-ulit sa mga files.

---

## 4. Agent Skills Suite (`agent-skills` by Addy Osmani)
- **Test-Driven Development (`test-driven-development`)**: Patunayan ang bugs o new features gamit ang tests bago baguhin ang code (Red-Green-Refactor).
- **Spec & Planning (`spec-driven-development`, `planning-and-task-breakdown`)**: Planuhin ang mga multi-step tasks at panatilihing maliit at incremental ang changes.
- **Code Simplification & Quality (`code-simplification`, `code-review-and-quality`)**: I-review ang diffs at linisin ang unnecessary complexity.
- **Debugging & Error Recovery (`debugging-and-error-recovery`)**: Systematic root-cause debugging sa halip na trial-and-error patching.

---

## 5. Git Commit & Push Guidelines
- **Huwag kailanman isama ang "Day X" o "Day 1/2/3" sa commit message**: Panatilihing malinis at concise (hal. `feat: implement data pipeline and baseline`).
- **Laging i-exclude ang skills (`.agents/`) at `GEMINI.md`**: Hindi dapat mai-push sa remote repo.

