"""
Comprehensive Comparative Analysis Module for Q&A Evaluator
============================================================

This module provides all functions needed for:
1. Model comparison analysis
2. Prompt version comparison
3. Model-prompt interaction analysis
4. Evidence-based decision rationale
5. Plotly visualizations

Usage:
    Import this module in your notebook and call the main analysis functions.
    All visualizations use Plotly. All conclusions are data-driven.
"""

import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional

def extract_response_text(response):
    if isinstance(response, dict) and 'content' in response:
        return response['content']
    elif hasattr(response, 'choices'):
        return response.choices[0].message.content
    elif isinstance(response, str):
        return response
    else:
        return str(response)

def clean_json_response(text):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()

def safe_json_parse(response):
    import json
    
    # Extract text
    if isinstance(response, dict) and 'content' in response:
        text = response['content']
    elif hasattr(response, 'choices'):
        text = response.choices[0].message.content
    elif isinstance(response, str):
        text = response
    else:
        text = str(response)
    
    # Remove markdown
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    # Parse
    return json.loads(text.strip())

# ================================================================
# SECTION 1: Model Comparison
# ================================================================

def compare_models(
    models: List[str],
    qa_db: List[Dict],
    prompt_template: str,
    evaluator_model: str,
    call_model_func,
    call_evaluator_func,
    n_questions: int = 10,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compare multiple models using a single prompt and evaluator.
    
    Args:
        models: List of model names to compare
        qa_db: Question-answer database
        prompt_template: The prompt template to use (should have .format() placeholders)
        evaluator_model: Model to use for evaluation
        call_model_func: Function to call student models
        call_evaluator_func: Function to call evaluator
        n_questions: Number of questions to test
        seed: Random seed for reproducibility
    
    Returns:
        (summary_df, detailed_results)
        - summary_df: DataFrame with mean, std, min, max scores per model
        - detailed_results: Dict mapping model names to list of individual results
    """
    
    print("="*60)
    print("MODEL COMPARISON ANALYSIS")
    print("="*60)
    
    # Sample questions
    np.random.seed(seed)
    sample_indices = np.random.choice(len(qa_db), size=min(n_questions, len(qa_db)), replace=False)
    test_questions = [qa_db[i] for i in sample_indices]
    
    print(f"\nTesting {len(models)} models on {len(test_questions)} questions")
    print(f"Evaluator: {evaluator_model}\n")
    
    # Storage
    model_results = {model: [] for model in models}
    
    # Test each model
    for model_name in models:
        print(f"\nTesting: {model_name}")
        print("-" * 40)
        
        for idx, qa in enumerate(test_questions):
            try:
                # Generate student answer
                student_answer = call_model_func(model_name, qa['question'])
                
                if not student_answer or "ERROR" in student_answer:
                    print(f"  Q{idx+1}: SKIP")
                    continue

                # Evaluate
                eval_response = call_evaluator_func(
                    evaluator_model,
                    prompt_template,
                    qa['question'],
                    qa['answer'],
                    student_answer
                )
                
                # Parse score
                try:
                    eval_json = safe_json_parse(eval_response)
                    score = eval_json.get('score_0_100', 0)
                    
                    model_results[model_name].append({
                        'question_id': idx,
                        'score': score,
                        'answer_length': len(student_answer),
                        'evaluation': eval_json
                    })
                    
                    print(f"  Q{idx+1}: {score:.0f}")
                    
                except json.JSONDecodeError:
                    print(f"  Q{idx+1}: PARSE_ERR")
                    continue
                
                time.sleep(0.5)  # Rate limit
                
            except Exception as e:
                print(f"  Q{idx+1}: ERROR - {str(e)[:30]}")
                continue
    
    # Calculate summary statistics
    summary_data = []
    for model_name, results in model_results.items():
        if len(results) > 0:
            scores = [r['score'] for r in results]
            summary_data.append({
                'model': model_name,
                'n_questions': len(results),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    return summary_df, model_results


def visualize_model_comparison(summary_df: pd.DataFrame, detailed_results: Dict):
    """
    Create comprehensive Plotly visualizations for model comparison.
    
    Args:
        summary_df: Summary DataFrame from compare_models()
        detailed_results: Detailed results dict from compare_models()
    """
    
    if len(summary_df) == 0:
        print("No data to visualize")
        return
    
    # Sort by performance
    df_sorted = summary_df.sort_values('mean_score', ascending=False)
    
    # Figure 1: 4-panel comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Mean Score by Model',
            'Consistency (Std Dev)',
            'Score Range',
            'Sample Size'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Panel 1: Mean Score
    fig.add_trace(
        go.Bar(
            x=df_sorted['model'],
            y=df_sorted['mean_score'],
            marker_color='rgb(55, 83, 109)',
            text=df_sorted['mean_score'].round(1),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Panel 2: Std Dev
    fig.add_trace(
        go.Bar(
            x=df_sorted['model'],
            y=df_sorted['std_score'],
            marker_color='rgb(26, 118, 255)',
            text=df_sorted['std_score'].round(1),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Panel 3: Score Range
    ranges = df_sorted['max_score'] - df_sorted['min_score']
    fig.add_trace(
        go.Bar(
            x=df_sorted['model'],
            y=ranges,
            marker_color='rgb(50, 171, 96)',
            text=[f"{mn:.0f}-{mx:.0f}" for mn, mx in zip(df_sorted['min_score'], df_sorted['max_score'])],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Panel 4: Sample Size
    fig.add_trace(
        go.Bar(
            x=df_sorted['model'],
            y=df_sorted['n_questions'],
            marker_color='rgb(219, 64, 82)',
            text=df_sorted['n_questions'],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="<b>Model Performance Comparison</b>",
        title_x=0.5,
        height=800
    )
    
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev", row=1, col=2)
    fig.update_yaxes(title_text="Range", row=2, col=1)
    fig.update_yaxes(title_text="N", row=2, col=2)
    
    fig.show()
    
    # Figure 2: Box plot distribution
    fig2 = go.Figure()
    
    for model_name in df_sorted['model']:
        scores = [r['score'] for r in detailed_results[model_name]]
        fig2.add_trace(go.Box(
            y=scores,
            name=model_name,
            boxmean='sd'
        ))
    
    fig2.update_layout(
        title="<b>Score Distribution by Model</b>",
        title_x=0.5,
        yaxis_title="Score (0-100)",
        xaxis_title="Model",
        height=500
    )
    
    fig2.show()


# ================================================================
# SECTION 2: Prompt Comparison
# ================================================================

def compare_prompts(
    prompt_versions: Dict[str, str],
    model_name: str,
    qa_db: List[Dict],
    evaluator_model: str,
    call_model_func,
    call_evaluator_func,
    n_questions: int = 10,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compare multiple prompt versions using a single model.
    
    Args:
        prompt_versions: Dict mapping prompt names to templates
        model_name: Model to use for generating answers
        qa_db: Question-answer database
        evaluator_model: Model to use for evaluation
        call_model_func: Function to call student model
        call_evaluator_func: Function to call evaluator
        n_questions: Number of questions to test
        seed: Random seed
    
    Returns:
        (summary_df, detailed_results)
    """
    
    print("="*60)
    print("PROMPT COMPARISON ANALYSIS")
    print("="*60)
    
    # Sample questions
    np.random.seed(seed)
    sample_indices = np.random.choice(len(qa_db), size=min(n_questions, len(qa_db)), replace=False)
    test_questions = [qa_db[i] for i in sample_indices]
    
    print(f"\nTesting {len(prompt_versions)} prompts on {len(test_questions)} questions")
    print(f"Model: {model_name}")
    print(f"Evaluator: {evaluator_model}\n")
    
    # Storage
    prompt_results = {pname: [] for pname in prompt_versions.keys()}
    
    # Test each prompt
    for prompt_name, prompt_template in prompt_versions.items():
        print(f"\nTesting: {prompt_name}")
        print("-" * 40)
        
        for idx, qa in enumerate(test_questions):
            try:
                # Generate student answer
                student_answer = call_model_func(model_name, qa['question'])
                
                if not student_answer or "ERROR" in student_answer:
                    print(f"  Q{idx+1}: SKIP")
                    continue
                
                # Evaluate with this prompt
                eval_response = call_evaluator_func(
                    evaluator_model,
                    prompt_template,
                    qa['question'],
                    qa['answer'],
                    student_answer
                )
                
                try:
                    eval_json = safe_json_parse(eval_response)
                    score = eval_json.get('model_judgment', {}).get('score_0_100', 0)
                    
                    prompt_results[prompt_name].append({
                        'question_id': idx,
                        'score': score,
                        'evaluation': eval_json
                    })
                    
                    print(f"  Q{idx+1}: {score:.0f}")
                    
                except json.JSONDecodeError:
                    print(f"  Q{idx+1}: PARSE_ERR")
                    continue
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Q{idx+1}: ERROR - {str(e)[:30]}")
                continue
    
    # Summary statistics
    summary_data = []
    for prompt_name, results in prompt_results.items():
        if len(results) > 0:
            scores = [r['score'] for r in results]
            summary_data.append({
                'prompt': prompt_name,
                'n_questions': len(results),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*60)
    print("PROMPT COMPARISON SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    return summary_df, prompt_results


def visualize_prompt_comparison(summary_df: pd.DataFrame, detailed_results: Dict, baseline: str = 'PROMPT_V1'):
    """
    Create Plotly visualizations for prompt comparison.
    
    Args:
        summary_df: Summary DataFrame from compare_prompts()
        detailed_results: Detailed results from compare_prompts()
        baseline: Name of baseline prompt for improvement calculation
    """
    
    if len(summary_df) == 0:
        print("No data to visualize")
        return
    
    df_sorted = summary_df.sort_values('prompt')
    
    # Figure 1: Mean Score and Consistency
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Mean Score by Prompt', 'Consistency (Std Dev)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    colors = ['rgb(230, 230, 250)', 'rgb(173, 216, 230)', 'rgb(135, 206, 250)', 'rgb(70, 130, 180)']
    
    fig.add_trace(
        go.Bar(
            x=df_sorted['prompt'],
            y=df_sorted['mean_score'],
            marker_color=colors[:len(df_sorted)],
            text=df_sorted['mean_score'].round(1),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=df_sorted['prompt'],
            y=df_sorted['std_score'],
            marker_color=colors[:len(df_sorted)],
            text=df_sorted['std_score'].round(1),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="<b>Prompt Version Comparison</b>",
        title_x=0.5,
        height=500
    )
    
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev", row=1, col=2)
    
    fig.show()
    
    # Figure 2: Distribution
    fig2 = go.Figure()
    
    for idx, prompt_name in enumerate(df_sorted['prompt']):
        scores = [r['score'] for r in detailed_results[prompt_name]]
        fig2.add_trace(go.Box(
            y=scores,
            name=prompt_name,
            marker_color=colors[idx],
            boxmean='sd'
        ))
    
    fig2.update_layout(
        title="<b>Score Distribution by Prompt</b>",
        title_x=0.5,
        yaxis_title="Score (0-100)",
        height=500
    )
    
    fig2.show()
    
    # Figure 3: Improvement over baseline
    if baseline in summary_df['prompt'].values:
        baseline_score = summary_df[summary_df['prompt'] == baseline]['mean_score'].values[0]
        
        improvements = []
        for _, row in df_sorted.iterrows():
            improvement = ((row['mean_score'] - baseline_score) / baseline_score) * 100
            improvements.append(improvement)
        
        fig3 = go.Figure(go.Bar(
            x=df_sorted['prompt'],
            y=improvements,
            marker_color=['rgb(100,100,100)' if x < 0 else 'rgb(50, 171, 96)' for x in improvements],
            text=[f"{x:+.1f}%" for x in improvements],
            textposition='outside'
        ))
        
        fig3.update_layout(
            title=f"<b>Improvement vs Baseline ({baseline})</b>",
            title_x=0.5,
            yaxis_title="Improvement (%)",
            height=400
        )
        
        fig3.add_hline(y=0, line_dash="dash", line_color="black")
        fig3.show()


# ================================================================
# SECTION 3: Model-Prompt Interaction
# ================================================================

def analyze_interaction(
    models: List[str],
    prompt_versions: Dict[str, str],
    qa_db: List[Dict],
    evaluator_model: str,
    call_model_func,
    call_evaluator_func,
    n_questions: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Analyze model-prompt interaction by testing all combinations.
    
    Returns:
        DataFrame with models as rows, prompts as columns, scores as values
    """
    
    print("="*60)
    print("MODEL-PROMPT INTERACTION ANALYSIS")
    print("="*60)
    
    # Sample questions
    np.random.seed(seed)
    sample_indices = np.random.choice(len(qa_db), size=min(n_questions, len(qa_db)), replace=False)
    test_questions = [qa_db[i] for i in sample_indices]
    
    print(f"\nTesting {len(models)} models × {len(prompt_versions)} prompts")
    print(f"Questions: {len(test_questions)}\n")
    
    # Storage: model -> prompt -> scores
    interaction_results = {
        model: {prompt: [] for prompt in prompt_versions.keys()}
        for model in models
    }
    
    total = len(models) * len(prompt_versions) * len(test_questions)
    completed = 0
    
    # Test all combinations
    for model_name in models:
        print(f"\n{model_name}")
        print("-" * 40)
        
        for prompt_name, prompt_template in prompt_versions.items():
            print(f"  {prompt_name}: ", end="", flush=True)
            
            for qa in test_questions:
                try:
                    student_answer = call_model_func(model_name, qa['question'])
                    
                    if not student_answer or "ERROR" in student_answer:
                        print("x", end="", flush=True)
                        continue
                    
                    eval_response = call_evaluator_func(
                        evaluator_model,
                        prompt_template,
                        qa['question'],
                        qa['answer'],
                        student_answer
                    )
                    
                    eval_json = safe_json_parse(eval_response)
                    score = eval_json.get('model_judgment', {}).get('score_0_100', 0)
                    
                    interaction_results[model_name][prompt_name].append(score)
                    completed += 1
                    print(".", end="", flush=True)
                    
                    time.sleep(0.5)
                    
                except Exception:
                    print("x", end="", flush=True)
                    continue
            
            print(f" ({completed}/{total})")
    
    # Create matrix
    matrix_data = []
    for model in models:
        row = [model]
        for prompt in sorted(prompt_versions.keys()):
            scores = interaction_results[model][prompt]
            mean_score = np.mean(scores) if len(scores) > 0 else 0
            row.append(mean_score)
        matrix_data.append(row)
    
    columns = ['Model'] + sorted(prompt_versions.keys())
    interaction_df = pd.DataFrame(matrix_data, columns=columns)
    
    print("\n" + "="*60)
    print("INTERACTION MATRIX")
    print("="*60)
    print(interaction_df.to_string(index=False))
    
    return interaction_df


def visualize_interaction(interaction_df: pd.DataFrame):
    """
    Create heatmap and grouped bar chart for interaction analysis.
    
    Args:
        interaction_df: DataFrame from analyze_interaction()
    """
    
    if len(interaction_df) == 0:
        print("No data to visualize")
        return
    
    models = interaction_df['Model'].tolist()
    prompt_cols = [c for c in interaction_df.columns if c != 'Model']
    z_data = interaction_df[prompt_cols].values
    
    # Figure 1: Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=prompt_cols,
        y=models,
        colorscale='RdYlGn',
        text=z_data.round(1),
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Score")
    ))
    
    fig.update_layout(
        title="<b>Model-Prompt Performance Matrix</b>",
        title_x=0.5,
        xaxis_title="Prompt Version",
        yaxis_title="Model",
        height=400 + len(models) * 50
    )
    
    fig.show()
    
    # Figure 2: Grouped bars
    fig2 = go.Figure()
    
    for prompt in prompt_cols:
        fig2.add_trace(go.Bar(
            name=prompt,
            x=models,
            y=interaction_df[prompt],
            text=interaction_df[prompt].round(1),
            textposition='outside'
        ))
    
    fig2.update_layout(
        title="<b>Model Performance Across Prompts</b>",
        title_x=0.5,
        xaxis_title="Model",
        yaxis_title="Mean Score",
        barmode='group',
        height=500,
        legend_title="Prompt"
    )
    
    fig2.show()
    
    # Find best combination
    best_score = 0
    best_combo = None
    
    for idx, row in interaction_df.iterrows():
        for prompt in prompt_cols:
            if row[prompt] > best_score:
                best_score = row[prompt]
                best_combo = (row['Model'], prompt)
    
    print("\n" + "="*60)
    print("BEST MODEL-PROMPT COMBINATION")
    print("="*60)
    if best_combo:
        print(f"Model: {best_combo[0]}")
        print(f"Prompt: {best_combo[1]}")
        print(f"Score: {best_score:.1f}")


# ================================================================
# SECTION 4: Evidence-Based Rationale
# ================================================================

def generate_rationale(
    model_df: Optional[pd.DataFrame] = None,
    prompt_df: Optional[pd.DataFrame] = None,
    interaction_df: Optional[pd.DataFrame] = None
):
    """
    Generate evidence-based decision rationale from analysis results.
    All conclusions derived strictly from provided data.
    
    Args:
        model_df: Model comparison summary
        prompt_df: Prompt comparison summary
        interaction_df: Interaction matrix
    """
    
    print("="*60)
    print("EVIDENCE-BASED DECISION RATIONALE")
    print("="*60)
    
    # 1. Model Selection Rationale
    if model_df is not None and len(model_df) > 0:
        print("\n" + "="*60)
        print("1. MODEL SELECTION RATIONALE")
        print("="*60)
        
        best_model_row = model_df.loc[model_df['mean_score'].idxmax()]
        
        print(f"\n✓ SELECTED MODEL: {best_model_row['model']}")
        print("\nData-driven reasons:")
        
        # Reason 1: Highest mean score
        print(f"\n  • Highest mean score: {best_model_row['mean_score']:.2f}")
        if len(model_df) > 1:
            second_best = model_df.nlargest(2, 'mean_score').iloc[1]['mean_score']
            diff = best_model_row['mean_score'] - second_best
            print(f"    - Outperforms second-best by {diff:.2f} points")
        
        # Reason 2: Consistency
        print(f"\n  • Consistency (Std Dev): {best_model_row['std_score']:.2f}")
        avg_std = model_df['std_score'].mean()
        if best_model_row['std_score'] < avg_std:
            print(f"    - Below average variability (avg={avg_std:.2f})")
            print(f"    - Indicates more reliable evaluations")
        else:
            print(f"    - Above average variability (avg={avg_std:.2f})")
        
        # Reason 3: Score range
        score_range = best_model_row['max_score'] - best_model_row['min_score']
        print(f"\n  • Score range: {best_model_row['min_score']:.0f}-{best_model_row['max_score']:.0f} (span={score_range:.0f})")
        
        # Reason 4: Coverage
        print(f"\n  • Evaluated on {best_model_row['n_questions']} questions")
    
    # 2. Prompt Selection Rationale
    if prompt_df is not None and len(prompt_df) > 0:
        print("\n\n" + "="*60)
        print("2. PROMPT SELECTION RATIONALE")
        print("="*60)
        
        best_prompt_row = prompt_df.loc[prompt_df['mean_score'].idxmax()]
        
        print(f"\n✓ SELECTED PROMPT: {best_prompt_row['prompt']}")
        print("\nData-driven reasons:")
        
        print(f"\n  • Highest mean score: {best_prompt_row['mean_score']:.2f}")
        
        if 'PROMPT_V1' in prompt_df['prompt'].values:
            v1_score = prompt_df[prompt_df['prompt'] == 'PROMPT_V1']['mean_score'].values[0]
            improvement = ((best_prompt_row['mean_score'] - v1_score) / v1_score) * 100
            print(f"    - Improvement over V1: {improvement:+.1f}%")
        
        print(f"\n  • Consistency (Std Dev): {best_prompt_row['std_score']:.2f}")
        
        print(f"\n  • Prompt evolution:")
        for _, row in prompt_df.sort_values('prompt').iterrows():
            print(f"    - {row['prompt']}: {row['mean_score']:.1f} (σ={row['std_score']:.1f})")
    
    # 3. Key Trade-offs
    print("\n\n" + "="*60)
    print("3. KEY TRADE-OFFS OBSERVED")
    print("="*60)
    print("\nFrom the data generated:")
    
    if model_df is not None and len(model_df) > 1:
        print("\n  Model trade-offs:")
        high_score = model_df.nlargest(2, 'mean_score')
        low_std = model_df.nsmallest(2, 'std_score')
        print(f"    • Highest scores: {', '.join(high_score['model'].tolist())}")
        print(f"    • Most consistent: {', '.join(low_std['model'].tolist())}")
    
    if interaction_df is not None and len(interaction_df) > 0:
        print("\n  Model-Prompt sensitivity:")
        prompt_cols = [c for c in interaction_df.columns if c != 'Model']
        for _, row in interaction_df.iterrows():
            scores = [row[p] for p in prompt_cols]
            score_range = max(scores) - min(scores)
            print(f"    • {row['Model']}: range = {score_range:.1f} pts")
    
    print("\n" + "="*60)
    print("All conclusions derived from notebook-generated data only.")
    print("="*60)


# ================================================================
# CONVENIENCE WRAPPER
# ================================================================

def run_full_analysis(
    models: List[str],
    prompt_versions: Dict[str, str],
    qa_db: List[Dict],
    evaluator_model: str,
    call_model_func,
    call_evaluator_func,
    n_questions_model: int = 10,
    n_questions_prompt: int = 10,
    n_questions_interaction: int = 5
):
    """
    Run complete comparative analysis: models, prompts, interaction, rationale.
    
    Returns:
        dict with all results and dataframes
    """
    
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARATIVE ANALYSIS")
    print("="*60)
    print(f"\nModels: {len(models)}")
    print(f"Prompts: {len(prompt_versions)}")
    print(f"Evaluator: {evaluator_model}\n")
    
    # Step 1: Model comparison
    print("\n### STEP 1: Model Comparison ###\n")
    model_df, model_details = compare_models(
        models, qa_db, list(prompt_versions.values())[0],  # Use first prompt
        evaluator_model, call_model_func, call_evaluator_func,
        n_questions=n_questions_model
    )
    visualize_model_comparison(model_df, model_details)
    
    # Step 2: Prompt comparison (using best model)
    print("\n### STEP 2: Prompt Comparison ###\n")
    best_model = model_df.loc[model_df['mean_score'].idxmax(), 'model'] if len(model_df) > 0 else models[0]
    prompt_df, prompt_details = compare_prompts(
        prompt_versions, best_model, qa_db,
        evaluator_model, call_model_func, call_evaluator_func,
        n_questions=n_questions_prompt
    )
    visualize_prompt_comparison(prompt_df, prompt_details)
    
    # Step 3: Interaction analysis (top 3 models)
    print("\n### STEP 3: Model-Prompt Interaction ###\n")
    top_models = model_df.nlargest(min(3, len(model_df)), 'mean_score')['model'].tolist() if len(model_df) >= 3 else models
    interaction_df = analyze_interaction(
        top_models, prompt_versions, qa_db,
        evaluator_model, call_model_func, call_evaluator_func,
        n_questions=n_questions_interaction
    )
    visualize_interaction(interaction_df)
    
    # Step 4: Rationale
    print("\n### STEP 4: Evidence-Based Rationale ###\n")
    generate_rationale(model_df, prompt_df, interaction_df)
    
    return {
        'model_df': model_df,
        'model_details': model_details,
        'prompt_df': prompt_df,
        'prompt_details': prompt_details,
        'interaction_df': interaction_df,
        'best_model': model_df.loc[model_df['mean_score'].idxmax(), 'model'] if len(model_df) > 0 else None,
        'best_prompt': prompt_df.loc[prompt_df['mean_score'].idxmax(), 'prompt'] if len(prompt_df) > 0 else None
    }


# ================================================================
# Example usage (for notebook integration)
# ================================================================

"""
# In your notebook:

from comparative_analysis import run_full_analysis

# Define your models and prompts
models = ["gpt-4o-mini", "gpt-3.5-turbo"]
prompts = {
    'PROMPT_V1': \"\"\"...\"\"\",
    'PROMPT_V2': \"\"\"...\"\"\",
    'PROMPT_V3': \"\"\"...\"\"\",
    'PROMPT_V4': \"\"\"...\"\"\"
}

# Run full analysis
results = run_full_analysis(
    models=models,
    prompt_versions=prompts,
    qa_db=qa_db,
    evaluator_model="gpt-4o",
    call_model_func=your_model_function,
    call_evaluator_func=your_evaluator_function
)

# Access results
best_model = results['best_model']
best_prompt = results['best_prompt']
model_comparison = results['model_df']
"""

