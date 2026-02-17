---
name: marimo-notebook
description: |
  Author interactive reactive Python notebooks with marimo (NOT API/programmatic use).
  Covers interactive cell execution where variables are automatically global.
license: MIT
compatibility: opencode
metadata:
  audience: data-scientists
  category: data-science
  framework: marimo
---

# Marimo Interactive Notebook Authoring

For INTERACTIVE notebooks (normal usage), NOT Cell.run() API.

## What I do

- Author interactive marimo notebooks where cells auto-execute based on dependencies
- Implement reactive UI components (mo.ui.*) with proper global binding
- Manage notebook layout and visualization
- Prevent Jupyter-isms that break marimo's reactive model

## When to use me

Use when:
- Creating .py notebook files for interactive data exploration
- Building reactive dashboards/apps with marimo
- Converting Jupyter notebooks (.ipynb) to marimo

Ask clarifying questions if:
- User mentions "Cell.run()", "reusable cells", or "testing cells" (indicates API use case, different patterns)
- User wants to "pass variables between cells" with parameters/returns (indicates wrong mental model)

## Critical concept: Interactive vs API

**This skill covers INTERACTIVE notebooks only** - where you edit and run cells in marimo editor.

**Interactive notebook behavior** (what we cover):
- Variables assigned in cells are **automatically global** (visible to all cells)
- Use `_prefix` (e.g., `_temp`) for cell-local variables only
- Returns affect display and UI state, **NOT** variable sharing
- Cells reference variables by direct name, not parameters

**API behavior** (Cell.run() - NOT covered here):
- Returns control what variables are "exported" for programmatic reuse
- Used for testing cells or reusing in other notebooks
- Different patterns apply

## Rules for interactive notebooks

### NO_PRINT: Print Doesn't Display in Run Mode
**Condition**: Outputting data, debug info, or user-facing messages  
**Action**: Use mo.md(), mo.stat(), mo.callout(), return objects, or marimo output functions  
**Violation**: print() works in edit mode but fails silently (no output) in run/app mode; creates poor UX

```python
# WRONG - fails silently in marimo run/app mode
print(f"Training complete. Accuracy: {accuracy}")
print("Processing row", i)

# GOOD - displays properly in all modes
mo.md(f"**Training complete.** Accuracy: {accuracy:.2%}")
mo.callout(f"Processing {i} of {total}", kind="info")
# Or use bare expression - marimo displays it
f"Processed {i} rows"

# GOOD - for metrics/statistics
mo.stat(value=f"{accuracy:.2%}", label="Accuracy", caption="+5% from baseline")
```

**Note**: print() is fine for temporary debugging in edit mode, but must be replaced before deploying/sharing the notebook.

---

### UNDERSCORE_LOCAL: Cell-Local Variables (Use Correctly)
**Good use**: `_tmp`, `_i` for throwaway temps within a cell that aren't needed elsewhere  
**Bad use**: `_df1`, `_df2` to dodge "variable already defined" errors  
**Why**: The latter hides DAG conflicts instead of fixing them (refactor to meaningful unique names)

```python
# GOOD - true cell-local temporaries
def process():
    _tmp = []  # Temporary buffer, cell-local
    for _i in range(10):  # Loop counter, cell-local
        _tmp.append(_i * 2)
    result = sum(_tmp)  # Only result is global
    return result

# BAD - hacking around uniqueness constraint instead of fixing it
def cell_a():
    _df = load_model1()  # WRONG: using _ to avoid "df already defined"
    return _df

def cell_b():  
    _df = load_model2()  # WRONG: _df again - confusing!
    return _df

# GOOD - meaningful unique names
def model1_data():
    model1_df = load_model1()  # Clear, unique name
    return model1_df

def model2_data():
    model2_df = load_model2()  # Clear, unique name
    return model2_df
```

---

### VAR_AUTO_GLOBAL: Variables Are Automatically Global
**Condition**: Assigning a variable in a cell  
**Behavior**: Variable is automatically global and available to all other cells  
**Local scope**: Use `_prefix` (e.g., `_i`, `_temp`) for cell-local variables that shouldn't be shared  
**Display**: Use bare expressions (not return) for cell output - place the value/expression as the last line

**Example:**
```python
# GOOD: df is automatically global, bare expression for display
def load_data():
    df = pd.read_csv("data.csv")  # df is GLOBAL
    df.head()  # Bare expression as last line = DISPLAYED

def analyze():  # No parameters!
    df.describe()  # Bare expression as last line = DISPLAYED

# Use _prefix for locals
def process():
    _tmp = []  # Local only
    for _i in range(10):
        _tmp.append(_i)
    result = sum(_tmp)  # result is global
    result  # Bare expression = DISPLAYED (not return)
```

---

### NO_PARAMETER_SHARING: Cells Don't Take Parameters
**Condition**: Sharing data between cells in interactive mode  
**Action**: Reference variables directly by name. Never use function parameters for cell-to-cell data flow.  
**Violation**: Trying to pass data via `def cell_b(df):` parameters. This breaks the DAG/model.

**CRITICAL EXAMPLE - The #1 Mistake:**
```python
# WRONG: Jupyter/functional thinking - trying to pass as parameter
@app.cell
def load_data():
    df = pd.read_csv("data.csv")
    return df

@app.cell  
def analyze(df):  # WRONG: Don't declare parameters for sharing!
    return df.describe()

# RIGHT: Direct global reference
@app.cell
def load_data():
    df = pd.read_csv("data.csv")  # df is automatically global
    return df.head()

@app.cell
def analyze():  # RIGHT: No parameters
    return df.describe()  # Direct reference to global df
```

**Note:** marimo auto-manages internal parameters for dependency tracking, but you NEVER write them manually for data sharing.

---

### UI_GLOBAL: UI Elements Must Be Global
**Condition**: Creating interactive UI elements (mo.ui.*)  
**Action**: Assign to global variable (no _ prefix) so marimo can track reactivity  
**Violation**: Anonymous elements or _local assignment - UI works but won't trigger reactive updates

```python
# GOOD
def controls():
    slider = mo.ui.slider(1, 10)  # Global - reactive
    return slider

# WRONG - anonymous (works but not reactive)
def controls():
    mo.ui.slider(1, 10)  # Can't reference elsewhere

# WRONG - local (breaks reactivity)
def controls():
    _slider = mo.ui.slider(1, 10)  # Local - not tracked
    return _slider
```

---

### UI_VALUE_ACCESS: Reading UI Values
**Condition**: Using UI element values in other cells  
**Action**: Access via `.value` attribute in cells OTHER than the definition cell  
**Violation**: Reading `.value` in the same cell as definition breaks reactivity

```python
# GOOD - Define in one cell, read in another
# Cell 1:
def controls():
    slider = mo.ui.slider(1, 10)  # Define
    return slider  # Display

# Cell 2:
def display(slider):
    return f"Value: {slider.value}"  # Read in DIFFERENT cell - reactive!

# BAD - Reading in same cell breaks reactivity
def broken():
    slider = mo.ui.slider(1, 10)  # Define
    # This cell won't re-run when slider changes!
    return f"Value: {slider.value}"  # Read in SAME cell

# WRONG - Shows object repr instead of value
def display(slider):
    return f"Value: {slider}"  # Shows <marimo.ui.slider...>
```

**Why**: The defining cell doesn't re-run on UI interaction; only dependent cells do. marimo tracks which cells reference (but don't define) UI variables and re-runs those.

**Exception**: SQL cells return DataFrames directly (no .value)

---

### ON_CHANGE_VALUE: Callback Signatures
**Condition**: Providing on_change callbacks to UI elements  
**Action**: Callback receives the raw value (int, str, etc.), NOT dict  
**Violation**: Trying to use Jupyter pattern `change['new']` or accessing `.value` on the value

```python
# GOOD
def handle(new_val):  # Receives the actual value
    print(new_val)

mo.ui.slider(on_change=handle)

# WRONG (Jupyter pattern)
def handle(change):
    val = change['new']  # ERROR: change is the value, not dict

# WRONG (expecting element)
def handle(elem):
    val = elem.value  # ERROR: elem IS the value
```

---

### STOP_NOT_EXCEPTION: Graceful Halting
**Condition**: Pausing execution for user input  
**Action**: Use mo.stop(predicate, message)  
**Violation**: Using raise/exception shows red error UI instead of clean message

```python
# GOOD
def compute(btn):
    mo.stop(not btn.value, "Click run to start")
    return expensive_calc()

# WRONG
def compute(btn):
    if not btn.value:
        raise ValueError("Click first")  # Ugly error UI
```

---

### FORM_GATE_VALUE: Form Submission Handling
**Condition**: Using element.form() for batched input  
**Action**: Check for None - form.value is None until submit clicked  
**Violation**: Processing None or assuming real-time updates

```python
# GOOD
def processor(form):
    mo.stop(form.value is None, "Submit to process")
    return analyze(form.value)

# WRONG (processes None on first run)
def processor(form):
    return analyze(form.value)
```

---

### SQL_DIRECT_DF: SQL Output
**Condition**: Assigning mo.sql() result  
**Action**: Variable IS the DataFrame - no .value needed  
**Violation**: Trying to access .value on a DataFrame

```python
# GOOD
def query():
    df = mo.sql("SELECT * FROM table")  # df IS DataFrame
    return df.head()

# WRONG
def query():
    df = mo.sql("SELECT * FROM table")
    return df.value  # ERROR
```

---

### NO_MUTATION_ACROSS_CELLS: Avoid Cross-Cell Mutation
**Condition**: Modifying data structures created in other cells  
**Action**: Mutate in the same cell that defines the variable, or create new variables  
**Violation**: marimo doesn't track object mutations, leading to stale state

```python
# BAD
def create_df():
    df = pd.DataFrame({"x": [1, 2, 3]})
    return df

def add_column(df):  # Mutation across cells
    df["y"] = df["x"] * 2  # Won't trigger reactivity properly

# GOOD
def create_df():
    df = pd.DataFrame({"x": [1, 2, 3]})
    df["y"] = df["x"] * 2  # Mutate where defined
    return df
```

---

## Validation checklist

- [ ] Use bare expressions for cell output (not return statements)
- [ ] Variables referenced by direct name (no manual parameters)
- [ ] UI elements assigned to globals (no _ prefix)
- [ ] .value used on UI elements in DIFFERENT cells (not same cell as definition)
- [ ] No print() statements (use mo.md/mo.stat/bare expressions)
- [ ] No mutation across cells (mutate where defined)
- [ ] Cell-local temps use _ prefix (not to dodge name conflicts)
- [ ] mo.stop for conditional halts
- [ ] on_change callbacks take value (not dict)

## Common Python/Jupyter idioms that break

| Jupyter/Naive Python | Marimo Interactive |
|---------------------|-------------------|
| `def analyze(df):` with cell params | `def analyze():` direct reference |
| `return df` to "export" variables | Variables auto-global; bare expression for display |
| `_ =` throwaway in Jupyter | `_tmp`, `_i` locals or meaningful names |
| `print()` for output | `mo.md()`, `mo.stat()`, or return objects |
| Read `.value` in same cell as UI def | Define UI in one cell, read `.value` in another |

## Quick reference

```python
import marimo as mo

# NOT print() - use mo output functions
# NOT def func(df): parameters - use direct global refs

@app.cell
def setup():
    """Variables here are automatically global"""
    df = load_data()
    _secret = "local only"  # _prefix = cell-local
    df.describe()  # Bare expression = display

@app.cell
def controls():
    """UI elements must be global for reactivity"""
    slider = mo.ui.slider(1, 100)
    return slider  # Don't read .value here!

@app.cell
def viz(slider):  # marimo auto-manages params for UI deps
    """Reference globals directly, access UI via .value in DIFFERENT cell"""
    filtered = df[df.x > slider.value]  # df global, slider.value access
    return filtered.plot()

if __name__ == "__main__":
    app.run()
```

## References

- https://docs.marimo.io/getting_started/key_concepts/
- https://docs.marimo.io/guides/reactivity/
- https://docs.marimo.io/guides/interactivity/
