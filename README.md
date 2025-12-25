# SCIG-RSI: Runtime Code Modification Framework

**SCIG-RSI** is a monolithic Python system for runtime AST (Abstract Syntax Tree) manipulation and recursive optimization. It modifies its own source code logic and control parameters during execution without external dependencies.

### Technical Specifications
* **AST-Based Modification (L5):** Mechanically alters constants, operators, and control flow using Python's `ast` module.
* **Algorithm Generation (L6):** Synthesizes executable code via token-level AST construction (`GranularASTGenerator`).
* **Logic Repair:** Implements heuristic-based diagnosis (`SemanticDiagnoser`) to correct logic errors.
* **Optimization Loop:** Adjusts internal parameters through feedback-based control rules (`MetaBrain`).

### Usage
`python scig_engine.py --mode [level6|repair|scig-demo|benchmark]`

> **Technical Note:** This software utilizes `ast.parse` and `shutil.copy2` for hot-patching. Execution in an isolated environment (Docker/VM) is required to prevent unintended file system modifications.
