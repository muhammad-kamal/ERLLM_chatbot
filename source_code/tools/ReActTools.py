import re
from typing import Dict


class DocumentSearchTool:
    def __init__(self, documents: Dict):
        self.documents = documents

    def execute(self, query: str) -> str:
        try:
            terms = [t.lower() for t in query.split()]
            if not terms:
                return "Empty query"
            # exclude not interesting words from the searching terms
            stop_words = {'from', 'since','due','as','it','he','she','they','we','I','and', 'or', 'but', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'his', 'her', 'them', 'there', 'these', 'those', 'its', 'are','if', 'else','for', 'before', 'after', 'this', 'up', 'down', 'i', 'you' }
            terms = [term for term in terms if term not in stop_words]
            if not terms: 
                return "Query contains only common words"
            results = {}
            for doc_id, content in self.documents.items():
                content_lower = content.lower()
                doc_results = []
                for term in terms:
                    if term not in content_lower:
                        continue
                    # Find all matches of this term
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    matches = list(pattern.finditer(content))
                    # Keep 3 matches for this term
                    snippet_count = 0
                    for match in matches:
                        if snippet_count >= 1:
                            break
                        idx = match.start()
                        context_size = 150
                        start = max(0, idx - context_size)
                        end = min(len(content), idx + context_size)
                        snippet = content[start:end]
                        doc_results.append(f"Term '{term}': {snippet}")
                        snippet_count += 1
                if doc_results:
                    results[doc_id] = doc_results
            if not results:
                return "No matches"
            output_lines = []
            for doc_id, snippets in results.items():
                output_lines.append(f"--- Document: {doc_id} ---")
                output_lines.extend(snippets)
                output_lines.append("")
            return "\n".join(output_lines)
        except Exception as e:
            return f"Error: {e}"


class CalculatorTool:
    def execute(self, query: str, tool_input: str) -> str:
        try:
            numbers = re.findall(r'\d+', tool_input)
            if len(numbers) >= 2:
                a, b = int(numbers[0]), int(numbers[1])
                if "add" in query.lower() or "sum" in query.lower() or "+" in tool_input:
                    return f"Result: {a} + {b} = {a + b}"
                elif "subtract" in query.lower() or "minus" in query.lower() or "-" in tool_input:
                    return f"Result: {a} - {b} = {a - b}"
                elif "multiply" in query.lower() or "times" in query.lower() or "*" in tool_input:
                    return f"Result: {a} × {b} = {a * b}"
                elif "divide" in query.lower() or "/" in tool_input:
                    return f"Result: {a} ÷ {b} = {a / b if b != 0 else 'undefined'}"
            return "Could not perform calculation. Specify numbers and operation."
        except Exception as e:
            return f"Calculation error: {str(e)}"

