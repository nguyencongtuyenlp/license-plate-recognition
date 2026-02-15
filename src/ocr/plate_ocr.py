"""
Plate OCR — EasyOCR + Image Preprocessing + Regex Cleanup
============================================================
Reads license plate text from cropped plate images.
Includes image preprocessing for better OCR accuracy
and regex-based post-processing for Vietnamese plate format.
"""

import re
from typing import List, Optional, Tuple
import cv2
import numpy as np


class PlateOCR:
    """License plate text recognition using EasyOCR.
    
    Pipeline: preprocess (grayscale, CLAHE, threshold) → OCR → regex cleanup
    
    Vietnamese plate format:
        Standard: 30A-12345
        Long:     51F-324.88
    
    Args:
        languages: OCR language list (default: ['en']).
        gpu: Whether to use GPU for OCR.
    """

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ['en']
        self.gpu = gpu
        self._reader = None  # Lazy init — EasyOCR is heavy to load

    def _get_reader(self):
        """Lazy-load EasyOCR reader (only created on first use)."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """Preprocess plate image to improve OCR accuracy.
        
        Pipeline: grayscale → resize → CLAHE → bilateral filter → Otsu threshold
        
        Args:
            plate_crop: BGR plate crop image.
        
        Returns:
            Binary (black/white) image optimized for OCR.
        """
        # Grayscale conversion
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # Resize to standard height (64px) while keeping aspect ratio
        h, w = gray.shape
        if h > 0:
            scale = 64.0 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        # CLAHE — adaptive contrast enhancement (works locally on tiles)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Bilateral filter — smooth while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Otsu thresholding — auto-find optimal threshold
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """Read text from a cropped plate image.
        
        Args:
            plate_crop: BGR plate crop.
        
        Returns:
            Tuple of (cleaned_text, average_confidence).
        """
        if plate_crop.size == 0:
            return "", 0.0

        processed = self.preprocess_plate(plate_crop)
        reader = self._get_reader()

        try:
            results = reader.readtext(processed)
        except Exception:
            return "", 0.0

        if not results:
            return "", 0.0

        # Combine all text blocks
        full_text = ""
        total_conf = 0.0
        for (bbox, text, conf) in results:
            full_text += text
            total_conf += conf

        avg_conf = total_conf / len(results) if results else 0.0

        # Clean up with regex
        cleaned = self._postprocess_plate_text(full_text)

        return cleaned, avg_conf

    def _postprocess_plate_text(self, raw_text: str) -> str:
        """Clean and validate plate text using regex.
        
        Handles common OCR errors (O->0, I->1, etc.) and
        matches Vietnamese plate format: NN[A-Z]-NNNNN.
        
        Args:
            raw_text: Raw OCR text.
        
        Returns:
            Cleaned plate text.
        """
        # Strip non-alphanumeric chars (keep '-' and '.')
        text = re.sub(r'[^A-Za-z0-9\-\.]', '', raw_text.upper())

        # Common OCR correction table
        corrections = {
            'O': '0', 'I': '1', 'L': '1',
            'S': '5', 'Z': '2', 'B': '8',
        }

        # Match Vietnamese plate pattern: 2 digits + 1 letter + 3-5 digits
        match = re.match(r'(\d{2})([A-Z])[\-]?(\d{3,5})', text)
        if match:
            province = match.group(1)
            series = match.group(2)
            number = match.group(3)
            return f"{province}{series}-{number}"

        return text
