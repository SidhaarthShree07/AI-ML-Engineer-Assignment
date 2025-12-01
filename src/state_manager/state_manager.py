"""State Manager: Maintain structured reasoning traces and session state"""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any
from dataclasses import asdict
from src.models import Strategy, TrainingMetrics


class StateManager:
    """Maintain structured reasoning traces and session state"""
    
    def __init__(self, session_id: str, output_dir: str):
        """Initialize state manager with session ID
        
        Args:
            session_id: Unique session identifier
            output_dir: Directory for output files
        """
        self.session_id = session_id
        self.output_dir = output_dir
        self.history: List[Dict[str, Any]] = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def log_action(self, phase: str, action: str, input_data: dict, output_data: dict):
        """Log an action with timestamp and details
        
        Args:
            phase: Current execution phase
            action: Action being performed
            input_data: Input parameters
            output_data: Output results
        """
        entry = {
            "type": "action",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "action": action,
            "input": input_data,
            "output": output_data
        }
        self.history.append(entry)
        
    def log_decision(self, decision_point: str, options: List[str], 
                     selected: str, reasoning: str):
        """Log a decision with reasoning
        
        Args:
            decision_point: Description of decision point
            options: Available options
            selected: Selected option
            reasoning: Reasoning for selection
        """
        entry = {
            "type": "decision",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_point": decision_point,
            "options": options,
            "selected": selected,
            "reasoning": reasoning
        }
        self.history.append(entry)
        
    def log_improvement(self, iteration: int, strategy: Strategy, 
                       metrics: TrainingMetrics, analysis: str):
        """Log self-improvement iteration
        
        Args:
            iteration: Iteration number
            strategy: Strategy used
            metrics: Performance metrics
            analysis: Analysis of results
        """
        # Convert dataclasses to dicts for JSON serialization
        strategy_dict = asdict(strategy)
        metrics_dict = asdict(metrics)
        
        entry = {
            "type": "improvement",
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": iteration,
            "strategy": strategy_dict,
            "metrics": metrics_dict,
            "analysis": analysis
        }
        self.history.append(entry)
        
    def save_trace(self) -> str:
        """Save complete reasoning trace to JSON file
        
        Returns:
            Path to saved trace file
        """
        trace_data = {
            "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "history": self.history
        }
        
        trace_filename = f"reasoning_trace_{self.session_id}.json"
        trace_path = os.path.join(self.output_dir, trace_filename)
        
        with open(trace_path, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        return trace_path
        
    def get_history(self) -> List[dict]:
        """Retrieve execution history
        
        Returns:
            List of logged events
        """
        return self.history
