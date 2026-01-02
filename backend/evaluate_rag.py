import logging
from typing import List, Dict
import numpy as np
from backend.rag_pipeline import MedicalRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self):
        self.pipeline = MedicalRAGPipeline()
    
    def evaluate_retrieval_quality(self, queries: List[str]) -> Dict:
        """Evaluate retrieval metrics"""
        
        retrieval_scores = []
        num_results = []
        
        for query in queries:
            context = self.pipeline.retrieve_context(query)
            num_results.append(len(context))
            
            if context:
                avg_score = np.mean([c['similarity_score'] for c in context])
                retrieval_scores.append(avg_score)
        
        metrics = {
            "avg_retrieval_score": np.mean(retrieval_scores) if retrieval_scores else 0,
            "avg_num_results": np.mean(num_results),
            "min_score": np.min(retrieval_scores) if retrieval_scores else 0,
            "max_score": np.max(retrieval_scores) if retrieval_scores else 0
        }
        
        return metrics
    
    def evaluate_answer_quality(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate answer generation quality
        
        test_cases format:
        [
            {
                "query": "What is aspirin used for?",
                "expected_keywords": ["pain", "fever", "inflammation"]
            }
        ]
        """
        
        results = {
            "total_cases": len(test_cases),
            "successful_answers": 0,
            "keyword_match_rate": []
        }
        
        for case in test_cases:
            query = case["query"]
            expected_keywords = case.get("expected_keywords", [])
            
            answer, context, llm = self.pipeline.generate_answer(query)
            
            if answer and "❌" not in answer:
                results["successful_answers"] += 1
                
                # Check keyword presence
                if expected_keywords:
                    answer_lower = answer.lower()
                    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
                    match_rate = matches / len(expected_keywords)
                    results["keyword_match_rate"].append(match_rate)
        
        results["success_rate"] = results["successful_answers"] / results["total_cases"] if results["total_cases"] > 0 else 0
        if results["keyword_match_rate"]:
            results["avg_keyword_match"] = np.mean(results["keyword_match_rate"])
        
        return results
    
    def run_evaluation(self) -> None:
        """Run complete evaluation suite"""
        
        logger.info("=" * 60)
        logger.info("Starting RAG System Evaluation")
        logger.info("=" * 60)
        
        # Test queries
        test_queries = [
            "What are the side effects of aspirin?",
            "How does insulin work in diabetes treatment?",
            "What are the symptoms of hypertension?",
            "Explain the mechanism of action of antibiotics",
            "What are contraindications for beta blockers?"
        ]
        
        # Evaluate retrieval
        logger.info("\n Evaluating Retrieval Quality...")
        retrieval_metrics = self.evaluate_retrieval_quality(test_queries)
        
        print("\n🔍 Retrieval Metrics:")
        for key, value in retrieval_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Evaluate answers
        test_cases = [
            {
                "query": "What are the side effects of aspirin?",
                "expected_keywords": ["bleeding", "stomach", "gastrointestinal"]
            },
            {
                "query": "What is diabetes?",
                "expected_keywords": ["glucose", "insulin", "blood sugar"]
            }
        ]
        
        logger.info("\n Evaluating Answer Quality...")
        answer_metrics = self.evaluate_answer_quality(test_cases)
        
        print("\n💡 Answer Generation Metrics:")
        for key, value in answer_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Complete")
        logger.info("=" * 60)


def main():
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()