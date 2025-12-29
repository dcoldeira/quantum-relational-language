#!/usr/bin/env python3
"""
Quantum Advisor - The Honest Quantum Computing Consultant

Tells you whether quantum computing makes sense for your problem.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class QuantumViability(Enum):
    """Assessment of whether quantum makes sense"""
    PROVEN_ADVANTAGE = "proven_advantage"
    HEURISTIC_POTENTIAL = "heuristic_potential"
    NO_ADVANTAGE = "no_advantage"
    UNKNOWN = "unknown"


@dataclass
class Assessment:
    """Result of quantum viability assessment"""
    problem_class: str
    viable: bool
    viability_level: QuantumViability
    quantum_advantage: Optional[str]
    speedup_type: Optional[str]
    confidence_level: str

    # Resource requirements
    required_qubits: str
    achievability: str
    timeline: str

    # Comparisons
    classical_baseline: str
    breakeven_point: str

    # Recommendations
    recommendation: str
    when_to_use_quantum: str
    when_to_use_classical: str

    # Evidence
    citations: List[str]
    caveats: List[str]

    def summary(self) -> str:
        """Generate human-readable summary"""
        if self.viable:
            icon = "✓" if self.viability_level == QuantumViability.PROVEN_ADVANTAGE else "⚠"
            return f"""
{icon} QUANTUM {'RECOMMENDED' if self.viability_level == QuantumViability.PROVEN_ADVANTAGE else 'POSSIBLE'} for {self.problem_class}

QUANTUM ADVANTAGE: {self.quantum_advantage} ({self.confidence_level})
SPEEDUP TYPE: {self.speedup_type}

RESOURCE REQUIREMENTS:
  • Qubits: {self.required_qubits}
  • Achievability: {self.achievability}
  • Timeline: {self.timeline}

CLASSICAL BASELINE: {self.classical_baseline}
BREAKEVEN POINT: {self.breakeven_point}

RECOMMENDATION: {self.recommendation}

WHEN TO USE QUANTUM: {self.when_to_use_quantum}

WHEN TO USE CLASSICAL: {self.when_to_use_classical}

CAVEATS:
{self._format_list(self.caveats)}

CITATIONS: {', '.join(self.citations)}
"""
        else:
            return f"""
❌ QUANTUM NOT RECOMMENDED for {self.problem_class}

REASON: {self.quantum_advantage}
CONFIDENCE: {self.confidence_level}

CLASSICAL BASELINE: {self.classical_baseline}
PERFORMANCE: {self.breakeven_point}

RECOMMENDATION: {self.recommendation}

{self.when_to_use_classical}

CAVEATS:
{self._format_list(self.caveats)}

CITATIONS: {', '.join(self.citations)}
"""

    def _format_list(self, items: List[str]) -> str:
        """Format list with bullet points"""
        return '\n'.join(f"  • {item}" for item in items)


class QuantumAdvisor:
    """
    The honest quantum computing consultant.

    Provides reality checks on quantum viability for different problem classes.
    """

    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize advisor with knowledge base.

        Args:
            knowledge_base_path: Path to knowledge_base.json (default: same directory)
        """
        if knowledge_base_path is None:
            kb_path = Path(__file__).parent / "knowledge_base.json"
        else:
            kb_path = Path(knowledge_base_path)

        with open(kb_path, 'r') as f:
            self.kb = json.load(f)

        print(f"Loaded Quantum Advisor v{self.kb['version']}")
        print(f"Knowledge base: {len(self.kb['problem_classes'])} problem classes, "
              f"{len(self.kb['quantum_algorithms'])} algorithms")

    def assess(self, problem_description: str) -> Assessment:
        """
        Assess whether quantum computing makes sense for a problem.

        Args:
            problem_description: Natural language description or problem_class_id

        Returns:
            Assessment with honest evaluation and recommendations
        """
        # Simple classification: look for exact match or keyword match
        problem_class_id = self._classify_problem(problem_description)

        if not problem_class_id:
            return self._unknown_problem_assessment(problem_description)

        # Get problem class data
        pc = self.kb['problem_classes'][problem_class_id]

        # Determine viability
        if pc['quantum_status']['advantage_proven']:
            if pc['quantum_status']['confidence_level'] == 'proven':
                viability = QuantumViability.PROVEN_ADVANTAGE
                viable = True
            else:
                viability = QuantumViability.HEURISTIC_POTENTIAL
                viable = True
        else:
            viability = QuantumViability.NO_ADVANTAGE
            viable = False

        # Build assessment
        return Assessment(
            problem_class=pc['name'],
            viable=viable,
            viability_level=viability,
            quantum_advantage=self._describe_advantage(pc),
            speedup_type=pc['quantum_status']['advantage_type'],
            confidence_level=pc['quantum_status']['confidence_level'],

            required_qubits=pc['breakeven_analysis']['required_qubits'],
            achievability=pc['breakeven_analysis']['current_achievability'],
            timeline=pc['breakeven_analysis']['estimated_timeline'],

            classical_baseline=self._describe_classical(pc),
            breakeven_point=pc['breakeven_analysis']['problem_size_threshold'],

            recommendation=self._generate_recommendation(pc, viable),
            when_to_use_quantum=pc['recommendations']['when_to_use_quantum'],
            when_to_use_classical=pc['recommendations']['when_to_use_classical'],

            citations=self._format_citations(pc['citations']),
            caveats=pc['quantum_status']['caveats']
        )

    def list_known_problems(self) -> List[str]:
        """List all problem classes in knowledge base"""
        return [
            f"{pc['name']} ({pc_id}) - {pc['quantum_status']['advantage_type']} advantage"
            for pc_id, pc in self.kb['problem_classes'].items()
        ]

    def get_algorithm_details(self, algorithm_id: str) -> Dict:
        """Get detailed information about a quantum algorithm"""
        if algorithm_id not in self.kb['quantum_algorithms']:
            raise ValueError(f"Unknown algorithm: {algorithm_id}")

        return self.kb['quantum_algorithms'][algorithm_id]

    def compare_approaches(self, problem_class_id: str) -> str:
        """Compare quantum vs classical approaches for a problem"""
        if problem_class_id not in self.kb['problem_classes']:
            raise ValueError(f"Unknown problem class: {problem_class_id}")

        pc = self.kb['problem_classes'][problem_class_id]

        output = f"\n{'='*70}\n"
        output += f"QUANTUM vs CLASSICAL: {pc['name']}\n"
        output += f"{'='*70}\n\n"

        # Classical side
        output += "CLASSICAL APPROACH:\n"
        output += f"  Algorithm: {pc['classical_baseline']['best_known_algorithm']}\n"
        output += f"  Complexity: {pc['classical_baseline']['complexity']}\n"
        output += f"  Performance: {pc['classical_baseline']['practical_performance']}\n"
        output += f"  Tools: {', '.join(pc['classical_baseline']['state_of_art_tools'])}\n\n"

        # Quantum side
        output += "QUANTUM APPROACH:\n"
        for qa in pc['quantum_approaches']:
            algo = self.kb['quantum_algorithms'][qa['algorithm_id']]
            output += f"  Algorithm: {algo['name']}\n"
            output += f"  Complexity: {algo['speedup']['quantum_complexity']}\n"
            output += f"  Speedup: {algo['speedup']['type']}\n"
            output += f"  Status: {qa['status']}\n"
            output += f"  Qubits: {algo['resource_requirements']['qubits']['formula']}\n\n"

        # Breakeven
        output += "BREAKEVEN ANALYSIS:\n"
        output += f"  Threshold: {pc['breakeven_analysis']['problem_size_threshold']}\n"
        output += f"  Timeline: {pc['breakeven_analysis']['estimated_timeline']}\n"
        output += f"  Achievability: {pc['breakeven_analysis']['current_achievability']}\n\n"

        # Recommendation
        output += "RECOMMENDATION:\n"
        output += f"  {self._generate_recommendation(pc, pc['quantum_status']['advantage_proven'])}\n"

        return output

    def _classify_problem(self, description: str) -> Optional[str]:
        """Classify problem description to problem class ID"""
        description_lower = description.lower()

        # Direct match
        if description_lower in self.kb['problem_classes']:
            return description_lower

        # Keyword matching (simple for now)
        keywords = {
            'integer_factorization': ['factor', 'factorization', 'factoring', 'rsa', 'shor'],
            'unstructured_search': ['search', 'database', 'grover', 'find'],
            'traveling_salesman': ['tsp', 'traveling', 'salesman', 'route', 'routing'],
            'quantum_chemistry': ['chemistry', 'molecule', 'molecular', 'chemical', 'drug', 'vqe']
        }

        for pc_id, kw_list in keywords.items():
            if any(kw in description_lower for kw in kw_list):
                return pc_id

        return None

    def _unknown_problem_assessment(self, description: str) -> Assessment:
        """Generate assessment for unknown problem"""
        return Assessment(
            problem_class=description,
            viable=False,
            viability_level=QuantumViability.UNKNOWN,
            quantum_advantage="Unknown - not in knowledge base",
            speedup_type="unknown",
            confidence_level="unknown",

            required_qubits="Unknown",
            achievability="unknown",
            timeline="Unknown",

            classical_baseline="Unknown (check classical CS literature)",
            breakeven_point="Unknown",

            recommendation=f"No knowledge base entry for '{description}'. Consult quantum computing expert.",
            when_to_use_quantum="Unknown - problem not classified",
            when_to_use_classical="Default to classical unless you have expert guidance",

            citations=[],
            caveats=[
                "This problem is not in our curated knowledge base",
                "Most problems do NOT have quantum advantage",
                "Consult a quantum computing expert before assuming quantum helps",
                f"Known problem classes: {', '.join(self.kb['problem_classes'].keys())}"
            ]
        )

    def _describe_advantage(self, problem_class: Dict) -> str:
        """Describe the quantum advantage"""
        qs = problem_class['quantum_status']
        if qs['advantage_proven']:
            return f"{qs['advantage_type'].replace('_', ' ').title()} quantum advantage (proven)"
        else:
            return f"No proven quantum advantage ({qs['advantage_type']})"

    def _describe_classical(self, problem_class: Dict) -> str:
        """Describe classical baseline"""
        cb = problem_class['classical_baseline']
        return f"{cb['best_known_algorithm']} - {cb['complexity']}"

    def _generate_recommendation(self, problem_class: Dict, viable: bool) -> str:
        """Generate overall recommendation"""
        if viable:
            timeline = problem_class['breakeven_analysis']['estimated_timeline']
            achievability = problem_class['breakeven_analysis']['current_achievability']

            if achievability == 'achievable_now':
                return "Quantum approach viable for research; try on NISQ hardware"
            elif achievability == 'near_term_5yr':
                return "Quantum approach promising; prepare for near-term quantum advantage"
            else:
                return f"Quantum advantage proven but requires fault-tolerant era ({timeline})"
        else:
            return "Use classical algorithms; no quantum advantage known"

    def _format_citations(self, citation_ids: List[str]) -> List[str]:
        """Format citations for display"""
        formatted = []
        for cid in citation_ids:
            if cid in self.kb['citations']:
                cite = self.kb['citations'][cid]
                authors = cite['authors'][0] if len(cite['authors']) == 1 else f"{cite['authors'][0]} et al."
                formatted.append(f"{authors} ({cite['year']})")
        return formatted


def main():
    """Demo of QuantumAdvisor"""
    advisor = QuantumAdvisor()

    print("\n" + "="*70)
    print("QUANTUM ADVISOR - Demo")
    print("="*70)

    # Test cases
    test_problems = [
        "integer factorization",
        "traveling salesman problem",
        "quantum chemistry simulation",
        "search a database",
        "optimize delivery routes",  # Should map to TSP
        "protein folding"  # Unknown
    ]

    for problem in test_problems:
        print(f"\n{'='*70}")
        print(f"QUERY: {problem}")
        print('='*70)

        assessment = advisor.assess(problem)
        print(assessment.summary())

    print("\n" + "="*70)
    print("COMPARISON EXAMPLE")
    print("="*70)
    print(advisor.compare_approaches("integer_factorization"))


if __name__ == "__main__":
    main()
