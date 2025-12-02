"""
Reverse AI - Goal-First Backward Planning Algorithm

This is the core innovation of DAUR-AI: instead of forward planning from current state,
we start with the goal and work backwards to find the optimal path.

Key Innovation:
- Traditional AI: State → Actions → Next State → ... → Goal
- Reverse AI: Goal → Required State → Required Actions → ... → Current State

This approach is more efficient for complex problems and enables better resource allocation.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
import hashlib
import json


class ActionType(Enum):
    """Types of actions in the planning space"""
    COMPUTATION = "computation"
    DATA_TRANSFER = "data_transfer"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "inference"
    RESOURCE_ALLOCATION = "resource_allocation"


@dataclass
class State:
    """Represents a state in the planning space"""
    id: str
    data: Dict[str, Any]
    timestamp: int
    resources: Dict[str, float] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, State) and self.id == other.id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'data': self.data,
            'timestamp': self.timestamp,
            'resources': self.resources,
        }


@dataclass
class Action:
    """Represents an action that transforms one state to another"""
    id: str
    type: ActionType
    name: str
    cost: float
    duration: int
    preconditions: List[str] = field(default_factory=list)
    effects: Dict[str, Any] = field(default_factory=dict)
    required_resources: Dict[str, float] = field(default_factory=dict)
    
    def can_apply(self, state: State) -> bool:
        """Check if action can be applied to given state"""
        # Check preconditions
        for condition in self.preconditions:
            if not self._check_condition(condition, state):
                return False
        
        # Check resources
        for resource, amount in self.required_resources.items():
            if state.resources.get(resource, 0) < amount:
                return False
        
        return True
    
    def _check_condition(self, condition: str, state: State) -> bool:
        """Check if a condition is satisfied in the state"""
        # Simple condition checking (can be extended)
        if '=' in condition:
            key, value = condition.split('=')
            return state.data.get(key.strip()) == value.strip()
        return True
    
    def apply(self, state: State) -> State:
        """Apply action to state and return new state"""
        new_data = state.data.copy()
        new_data.update(self.effects)
        
        new_resources = state.resources.copy()
        for resource, amount in self.required_resources.items():
            new_resources[resource] = new_resources.get(resource, 0) - amount
        
        new_state = State(
            id=self._generate_state_id(state, self),
            data=new_data,
            timestamp=state.timestamp + self.duration,
            resources=new_resources,
        )
        
        return new_state
    
    @staticmethod
    def _generate_state_id(state: State, action: 'Action') -> str:
        """Generate unique ID for resulting state"""
        data = f"{state.id}:{action.id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class PlanNode:
    """Node in the backward planning tree"""
    state: State
    action: Optional[Action]
    parent: Optional['PlanNode']
    g_cost: float  # Cost from goal to this node
    h_cost: float  # Heuristic cost from this node to start
    
    @property
    def f_cost(self) -> float:
        """Total cost (g + h)"""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def get_path(self) -> List[Tuple[State, Optional[Action]]]:
        """Get path from start to goal"""
        path = []
        current = self
        while current is not None:
            path.append((current.state, current.action))
            current = current.parent
        return list(reversed(path))


class ReverseAI:
    """
    Reverse AI Planning Engine
    
    Uses backward search from goal to current state to find optimal plans.
    """
    
    def __init__(self):
        self.actions: List[Action] = []
        self.explored: Set[str] = set()
    
    def register_action(self, action: Action) -> None:
        """Register an available action"""
        self.actions.append(action)
    
    def plan(
        self,
        current_state: State,
        goal_state: State,
        max_iterations: int = 10000,
    ) -> Optional[List[Tuple[State, Action]]]:
        """
        Create a plan from current state to goal state using backward search.
        
        Args:
            current_state: Starting state
            goal_state: Desired goal state
            max_iterations: Maximum search iterations
            
        Returns:
            List of (state, action) pairs representing the plan, or None if no plan found
        """
        self.explored.clear()
        
        # Priority queue for A* search (backward from goal)
        open_set: List[PlanNode] = []
        
        # Start from goal
        start_node = PlanNode(
            state=goal_state,
            action=None,
            parent=None,
            g_cost=0,
            h_cost=self._heuristic(goal_state, current_state),
        )
        
        heapq.heappush(open_set, start_node)
        
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current_node = heapq.heappop(open_set)
            
            # Check if we reached the current state (working backward)
            if self._is_goal_reached(current_node.state, current_state):
                plan = current_node.get_path()
                # Remove the first element (goal state with no action)
                return plan[1:] if len(plan) > 1 else []
            
            # Mark as explored
            self.explored.add(current_node.state.id)
            
            # Expand node (find actions that could lead to this state)
            for action in self._get_applicable_reverse_actions(current_node.state):
                # Apply action in reverse to get predecessor state
                predecessor_state = self._reverse_apply(current_node.state, action)
                
                if predecessor_state.id in self.explored:
                    continue
                
                g_cost = current_node.g_cost + action.cost
                h_cost = self._heuristic(predecessor_state, current_state)
                
                new_node = PlanNode(
                    state=predecessor_state,
                    action=action,
                    parent=current_node,
                    g_cost=g_cost,
                    h_cost=h_cost,
                )
                
                heapq.heappush(open_set, new_node)
        
        # No plan found
        return None
    
    def _get_applicable_reverse_actions(self, state: State) -> List[Action]:
        """
        Get actions that could have led to this state (reverse reasoning).
        In reverse planning, we look for actions whose effects match current state.
        """
        applicable = []
        
        for action in self.actions:
            # Check if action's effects are present in current state
            if self._effects_match(action, state):
                applicable.append(action)
        
        return applicable
    
    def _effects_match(self, action: Action, state: State) -> bool:
        """Check if action's effects are present in the state"""
        for key, value in action.effects.items():
            if state.data.get(key) != value:
                return False
        return True
    
    def _reverse_apply(self, state: State, action: Action) -> State:
        """
        Apply action in reverse: remove effects and restore preconditions.
        This gives us the state that must have existed before this action.
        """
        new_data = state.data.copy()
        
        # Remove effects
        for key in action.effects.keys():
            if key in new_data:
                del new_data[key]
        
        # Restore preconditions
        for condition in action.preconditions:
            if '=' in condition:
                key, value = condition.split('=')
                new_data[key.strip()] = value.strip()
        
        # Restore resources
        new_resources = state.resources.copy()
        for resource, amount in action.required_resources.items():
            new_resources[resource] = new_resources.get(resource, 0) + amount
        
        predecessor_state = State(
            id=self._generate_predecessor_id(state, action),
            data=new_data,
            timestamp=state.timestamp - action.duration,
            resources=new_resources,
        )
        
        return predecessor_state
    
    def _generate_predecessor_id(self, state: State, action: Action) -> str:
        """Generate ID for predecessor state"""
        data = f"{state.id}:reverse:{action.id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _heuristic(self, state1: State, state2: State) -> float:
        """
        Heuristic function for A* search.
        Estimates cost from state1 to state2.
        """
        # Simple heuristic: count differences in data
        differences = 0
        all_keys = set(state1.data.keys()) | set(state2.data.keys())
        
        for key in all_keys:
            if state1.data.get(key) != state2.data.get(key):
                differences += 1
        
        # Add resource differences
        all_resources = set(state1.resources.keys()) | set(state2.resources.keys())
        for resource in all_resources:
            diff = abs(state1.resources.get(resource, 0) - state2.resources.get(resource, 0))
            differences += diff * 0.1  # Weight resource differences less
        
        return float(differences)
    
    def _is_goal_reached(self, state1: State, state2: State) -> bool:
        """Check if two states are equivalent (goal reached)"""
        # Check data equality
        if state1.data != state2.data:
            return False
        
        # Check resource sufficiency
        for resource, amount in state2.resources.items():
            if state1.resources.get(resource, 0) < amount:
                return False
        
        return True
    
    def optimize_plan(
        self,
        plan: List[Tuple[State, Action]],
    ) -> List[Tuple[State, Action]]:
        """
        Optimize a plan by removing redundant actions.
        """
        if not plan:
            return plan
        
        optimized = []
        i = 0
        
        while i < len(plan):
            state, action = plan[i]
            
            # Check if we can skip this action
            can_skip = False
            if i + 1 < len(plan):
                next_state, next_action = plan[i + 1]
                # If next action can be applied directly to current state
                if next_action and next_action.can_apply(state):
                    can_skip = True
            
            if not can_skip:
                optimized.append((state, action))
            
            i += 1
        
        return optimized
    
    def to_json(self, plan: List[Tuple[State, Action]]) -> str:
        """Convert plan to JSON format"""
        plan_data = []
        for state, action in plan:
            plan_data.append({
                'state': state.to_dict(),
                'action': {
                    'id': action.id,
                    'type': action.type.value,
                    'name': action.name,
                    'cost': action.cost,
                    'duration': action.duration,
                } if action else None,
            })
        
        return json.dumps(plan_data, indent=2)


# Example usage and testing
if __name__ == '__main__':
    # Create Reverse AI planner
    planner = ReverseAI()
    
    # Register available actions
    planner.register_action(Action(
        id='train_model',
        type=ActionType.MODEL_TRAINING,
        name='Train AI Model',
        cost=100.0,
        duration=3600,
        preconditions=['data_available=true'],
        effects={'model_trained': 'true'},
        required_resources={'gpu': 1.0, 'memory': 4096},
    ))
    
    planner.register_action(Action(
        id='collect_data',
        type=ActionType.DATA_TRANSFER,
        name='Collect Training Data',
        cost=50.0,
        duration=1800,
        preconditions=[],
        effects={'data_available': 'true'},
        required_resources={'network': 100},
    ))
    
    planner.register_action(Action(
        id='run_inference',
        type=ActionType.MODEL_INFERENCE,
        name='Run Model Inference',
        cost=10.0,
        duration=60,
        preconditions=['model_trained=true'],
        effects={'prediction': 'available'},
        required_resources={'gpu': 0.5},
    ))
    
    # Define current state
    current_state = State(
        id='start',
        data={},
        timestamp=0,
        resources={'gpu': 2.0, 'memory': 8192, 'network': 1000},
    )
    
    # Define goal state
    goal_state = State(
        id='goal',
        data={'prediction': 'available'},
        timestamp=5460,  # 1h 31min
        resources={},
    )
    
    # Create plan
    print("Planning from current state to goal using Reverse AI...")
    plan = planner.plan(current_state, goal_state)
    
    if plan:
        print(f"\n✅ Plan found with {len(plan)} steps:\n")
        for i, (state, action) in enumerate(plan, 1):
            if action:
                print(f"{i}. {action.name}")
                print(f"   Cost: {action.cost}, Duration: {action.duration}s")
                print(f"   State: {state.data}")
        
        print(f"\n📊 Plan JSON:\n{planner.to_json(plan)}")
    else:
        print("❌ No plan found")
