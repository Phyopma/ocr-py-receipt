import re
from typing import Dict, List, Tuple


class TextProcessor:
    def __init__(self):
        # Initialize keyword patterns for document classification
        self.receipt_patterns = {
            'keywords': ['total', 'subtotal', 'tax', 'tip', 'merchant', 'store', 'change due', 'cash', 'card',
                         'payment', 'balance', 'amount', 'receipt', 'transaction', 'sale', 'terminal', 'register'],
            'price_pattern': r'\$?\d+\.\d{2}',  # Price format like $10.99
            'date_pattern': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Date formats
        }

        self.invoice_patterns = {
            'keywords': ['invoice', 'due date', 'amount due', 'billing', 'invoice number', 'account',
                         'payment terms', 'bill to', 'ship to', 'po number', 'order number', 'net', 'terms'],
            # Invoice number patterns
            'invoice_num_pattern': r'inv[oice]*[^a-zA-Z0-9]?\s*#?\s*\d+',
            'date_pattern': r'due\s+date|payment\s+due',  # Due date related patterns
        }

    def clean_text(self, text: str) -> str:
        """Clean OCR output with essential receipt text processing.

        Args:
            text (str): Raw OCR text

        Returns:
            str: Cleaned text
        """
        # Preserve line breaks and strip whitespace
        text = '\n'.join(line.strip() for line in text.split('\n'))

        # Fix common OCR errors in numbers
        text = re.sub(r'\bI\b', '1', text)  # Fix I -> 1
        text = re.sub(r'\bO\b', '0', text)  # Fix O -> 0
        text = re.sub(r'\bB\b', '8', text)  # Fix B -> 8

        # Fix price formats
        # Fix decimal spacing
        text = re.sub(r'(\d+)\s*\.\s*(\d{2})', r'\1.\2', text)
        # Fix missing decimal point
        text = re.sub(r'(\d+)\s+(\d{2})(?=\s|$)', r'\1.\2', text)

        return text.strip()

    def _fix_receipt_formatting(self, text: str) -> str:
        """Fix common receipt formatting issues.

        Args:
            text (str): Text with potential formatting issues

        Returns:
            str: Text with improved formatting
        """
        # Fix common item separator patterns
        text = re.sub(r'[-_=]{3,}', '\n', text)

        # Fix common quantity patterns (e.g., "2 x" -> "2x")
        text = re.sub(r'(\d+)\s*[xX]\s*', r'\1x ', text)

        # Fix common total line patterns
        text = re.sub(r'(?i)(sub)?total\s*:', r'\1TOTAL:', text)

        # Normalize multiple line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors and artifacts.

        Args:
            text (str): Text with potential OCR errors

        Returns:
            str: Text with common errors fixed
        """
        # Common OCR substitutions with context awareness
        substitutions = [
            (r'0', 'O'),  # Zero to O confusion
            (r'1', 'I'),  # One to I confusion
            (r'5', 'S'),  # Five to S confusion
            (r'8', 'B'),  # Eight to B confusion
            (r'rn', 'm'),  # 'rn' to 'm' confusion
            (r'cl', 'd'),  # 'cl' to 'd' confusion
            (r'vv', 'w'),  # 'vv' to 'w' confusion
        ]

        # Fix common spacing issues around punctuation
        text = re.sub(r'\s*([.,!?:;])\s*', r'\1 ', text)

        # Fix broken words (e.g., 'T otal' -> 'Total')
        text = re.sub(r'([A-Z])\s+([a-z]{1,2})\b', r'\1\2', text)

        # Apply substitutions with context awareness
        for pattern, replacement in substitutions:
            # Only substitute when the pattern appears in a likely error context
            if pattern.isdigit():
                # For digit substitutions, only apply when isolated
                text = re.sub(f'(?<!\d){pattern}(?!\d)', replacement, text)
            else:
                # For character pattern substitutions
                text = re.sub(pattern, replacement, text)

        # Fix common currency symbol misrecognitions
        text = re.sub(r'\$5', '$S', text)  # Fix $5 -> $S confusion
        text = re.sub(r'\$0', '$O', text)  # Fix $0 -> $O confusion

        return text

    def classify_document(self, text: str) -> Tuple[str, float]:
        """Classify document type with confidence score.

        Args:
            text (str): Cleaned document text

        Returns:
            Tuple[str, float]: Document type and confidence score
        """
        text = text.lower()

        # Calculate receipt characteristics
        receipt_score = self._calculate_receipt_score(text)

        # Calculate invoice characteristics
        invoice_score = self._calculate_invoice_score(text)

        # Normalize scores
        total_score = receipt_score + invoice_score
        if total_score > 0:
            receipt_confidence = receipt_score / total_score
            invoice_confidence = invoice_score / total_score
        else:
            receipt_confidence = invoice_confidence = 0

        # Make classification decision
        if receipt_score >= 2 and receipt_confidence > 0.6:
            return 'receipt', receipt_confidence
        elif invoice_score >= 2 and invoice_confidence > 0.6:
            return 'invoice', invoice_confidence
        else:
            return 'other', max(receipt_confidence, invoice_confidence)

    def _calculate_receipt_score(self, text: str) -> float:
        """Calculate receipt characteristics score.

        Args:
            text (str): Preprocessed text

        Returns:
            float: Receipt score
        """
        score = 0.0

        # Check for receipt keywords
        keyword_matches = sum(
            keyword in text for keyword in self.receipt_patterns['keywords'])
        score += keyword_matches * 0.5

        # Check for price patterns
        price_matches = len(re.findall(
            self.receipt_patterns['price_pattern'], text))
        score += min(price_matches / 3, 2.0)  # Cap the price pattern score

        # Check for date patterns
        if re.search(self.receipt_patterns['date_pattern'], text):
            score += 1.0

        return score

    def _calculate_invoice_score(self, text: str) -> float:
        """Calculate invoice characteristics score.

        Args:
            text (str): Preprocessed text

        Returns:
            float: Invoice score
        """
        score = 0.0

        # Check for invoice keywords
        keyword_matches = sum(
            keyword in text for keyword in self.invoice_patterns['keywords'])
        score += keyword_matches * 0.5

        # Check for invoice number pattern
        if re.search(self.invoice_patterns['invoice_num_pattern'], text, re.IGNORECASE):
            score += 2.0

        # Check for due date pattern
        if re.search(self.invoice_patterns['date_pattern'], text, re.IGNORECASE):
            score += 1.0

        return score
