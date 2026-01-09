# Multi-Model Support Guide

This guide explains how to configure and use the multi-model feature in the LEGO Lean Production simulation.

## Overview

The system now supports **4 different model types** (M1, M2, M3, M4), each with potentially different material requirements at different stages. This allows you to simulate production lines that produce multiple product variants.

## Key Features

1. **Model-specific material requirements**: Each stage can have different material requirements for different models
2. **Model-specific outputs**: Different models can produce different finished goods
3. **Model tracking**: Each order/job is tagged with a model_id that follows it through the production process
4. **Backward compatibility**: If no model_id is specified, the system uses default materials (legacy behavior)

## Configuration

### 1. Define Models

In your `CONFIG` dictionary, add a `models` section:

```python
"models": {
    "M1": {
        "name": "Model 1 - Standard Glider",
        "description": "Standard configuration with default materials"
    },
    "M2": {
        "name": "Model 2 - Premium Glider",
        "description": "Premium model with additional materials"
    },
    "M3": {
        "name": "Model 3 - Basic Glider",
        "description": "Basic model with reduced materials"
    },
    "M4": {
        "name": "Model 4 - Custom Glider",
        "description": "Custom model with unique material requirements"
    }
}
```

### 2. Configure Model-Specific Materials per Stage

For each stage where you want model-specific behavior, add:

- `required_materials_by_model`: Dictionary mapping model_id to material requirements
- `output_buffers_by_model`: Dictionary mapping model_id to output specifications

**Example (Final Assembly Stage S5):**

```python
{
    "stage_id": "S5",
    "name": "Final Assembly",
    "team_id": "T5",
    "input_buffers": ["C3", "D1", "D2"],
    
    # Default materials (used if model_id is "default" or not found)
    "required_materials": {"bun03": 1, "saa01": 1, "sac01": 1},
    
    # Model-specific materials (overrides required_materials when model_id matches)
    "required_materials_by_model": {
        "M1": {"bun03": 1, "saa01": 1, "sac01": 1},  # Standard
        "M2": {"bun03": 2, "saa01": 1, "sac01": 1},  # Premium: needs extra bun03
        "M3": {"bun03": 1, "saa01": 1},               # Basic: no sac01
        "M4": {"bun03": 1, "saa01": 2, "sac01": 1}    # Custom: double saa01
    },
    
    # Default outputs
    "output_buffers": {"E": {"fg01": 1}},
    
    # Model-specific outputs (different models produce different finished goods)
    "output_buffers_by_model": {
        "M1": {"E": {"fg01": 1}},  # Standard glider
        "M2": {"E": {"fg02": 1}},  # Premium glider
        "M3": {"E": {"fg03": 1}},  # Basic glider
        "M4": {"E": {"fg04": 1}}   # Custom glider
    },
    
    # ... other stage parameters
}
```

### 3. Set Default Model

In `parameters`, specify the default model:

```python
"parameters": {
    "default_model_id": "M1",  # Default model if not specified
    # ... other parameters
}
```

## Usage

### Programmatic Usage

When releasing orders, specify the model type:

```python
from env import LegoLeanEnv, CONFIG

env = LegoLeanEnv(CONFIG, time_unit="sec", seed=42)

# Release 10 orders of Model 1
env.enqueue_orders(qty=10, model_id="M1")

# Release 5 orders of Model 2
env.enqueue_orders(qty=5, model_id="M2")

# Release without specifying model (uses default_model_id)
env.enqueue_orders(qty=3)  # Will use "M1" if default_model_id="M1"
```

### UI Usage (Streamlit)

1. Run the Streamlit app: `streamlit run app.py`
2. In the sidebar, you'll see a **"Model Type Selection"** section
3. Select the model type you want to produce
4. Set the initial release quantity
5. Click "Run Simulation"

The selected model type will be used for all orders released in that simulation run.

## How It Works

1. **Order Release**: When `enqueue_orders(qty, model_id)` is called, the model_id is stored in a queue for each release stage.

2. **Job Start**: When a stage starts processing:
   - For release stages: model_id is retrieved from the queue
   - For downstream stages: model_id is passed through events from upstream stages
   - If model_id is not available, the default model is used

3. **Material Selection**: 
   - The system first checks `required_materials_by_model[model_id]`
   - If found, uses those materials
   - Otherwise, falls back to `required_materials` (default/legacy)

4. **Output Selection**:
   - Similar to materials, checks `output_buffers_by_model[model_id]` first
   - Falls back to `output_buffers` if not found

5. **Model Tracking**: 
   - Each job gets a unique `job_id`
   - The `job_id` is mapped to `model_id` in `_job_model_map`
   - This allows model_id to be retrieved even if not explicitly passed in events

## Partial Model Support

You don't need to define materials for all models at all stages. The system will:

- Use model-specific materials if defined for that model at that stage
- Fall back to default materials if not defined
- This allows you to have some stages with model-specific behavior and others with shared behavior

**Example**: If S1 (Type Sorting) uses the same materials for all models, you only need to define `required_materials`. But if S5 (Final Assembly) has different requirements per model, define `required_materials_by_model`.

## Best Practices

1. **Start Simple**: Begin with one model (M1) and default materials. Add model-specific configurations gradually.

2. **Consistent Naming**: Use consistent model IDs (M1, M2, M3, M4) across all stages.

3. **Document Differences**: In your model definitions, clearly document what makes each model different.

4. **Test Incrementally**: Test each model separately before mixing them in production runs.

5. **Buffer Considerations**: Different models may produce different items. Make sure your buffers can handle all item types that will be produced.

## Troubleshooting

**Issue**: Model-specific materials not being used
- **Check**: Ensure `required_materials_by_model` is defined in the stage config
- **Check**: Verify the model_id matches exactly (case-sensitive)
- **Check**: Ensure model_id is being passed correctly through `enqueue_orders()`

**Issue**: Default materials always used
- **Check**: Verify `default_model_id` is set in parameters
- **Check**: Ensure models are defined in the `models` section of CONFIG

**Issue**: Model_id not propagating to downstream stages
- **Check**: The system passes model_id through events. If a stage doesn't receive model_id, it uses the default.
- **Note**: In complex routing scenarios, you may need to track model_id per buffer item (advanced feature).

## Example: Complete Multi-Model Configuration

See `env.py` CONFIG section for a complete example with:
- 4 model definitions (M1-M4)
- Model-specific materials at S5 (Final Assembly)
- Model-specific outputs producing different finished goods (fg01-fg04)
