"""Sequence-to-sequence strategy template with T5 and Text Normalization support"""

# =============================================================================
# TEXT NORMALIZATION TEMPLATE - Hybrid Rule-Based + T5 Approach
# For text-normalization-challenge-english-language and similar datasets
# =============================================================================

TEXT_NORMALIZATION_TEMPLATE = """
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# RULE-BASED NORMALIZERS FOR COMMON CLASSES
# These handle ~70% of cases efficiently without ML
# ============================================================================

class RuleBasedNormalizer:
    \"\"\"Rule-based text normalizer for common transformation classes.\"\"\"
    
    # Number words for cardinals and ordinals
    ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
            'seventeen', 'eighteen', 'nineteen']
    TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    ORDINAL_ONES = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
                   'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth',
                   'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth']
    ORDINAL_TENS = ['', '', 'twentieth', 'thirtieth', 'fortieth', 'fiftieth',
                   'sixtieth', 'seventieth', 'eightieth', 'ninetieth']
    
    MONTHS = {{1: 'january', 2: 'february', 3: 'march', 4: 'april', 5: 'may', 6: 'june',
              7: 'july', 8: 'august', 9: 'september', 10: 'october', 11: 'november', 12: 'december'}}
    
    @classmethod
    def normalize(cls, text, semiotic_class):
        \"\"\"Normalize text based on its semiotic class.\"\"\"
        text = str(text).strip()
        semiotic_class = str(semiotic_class).upper()
        
        # VERBATIM and PLAIN - keep as is (lowercase)
        if semiotic_class in ['VERBATIM', 'PLAIN', '<SELF>']:
            return text.lower()
        
        # PUNCT - expand punctuation
        if semiotic_class == 'PUNCT':
            return cls._normalize_punct(text)
        
        # CARDINAL - numbers to words
        if semiotic_class == 'CARDINAL':
            return cls._normalize_cardinal(text)
        
        # ORDINAL - ordinal numbers
        if semiotic_class == 'ORDINAL':
            return cls._normalize_ordinal(text)
        
        # DATE - date normalization
        if semiotic_class == 'DATE':
            return cls._normalize_date(text)
        
        # TIME - time normalization
        if semiotic_class == 'TIME':
            return cls._normalize_time(text)
        
        # MEASURE - measurements
        if semiotic_class == 'MEASURE':
            return cls._normalize_measure(text)
        
        # MONEY - currency
        if semiotic_class == 'MONEY':
            return cls._normalize_money(text)
        
        # TELEPHONE - phone numbers
        if semiotic_class == 'TELEPHONE':
            return cls._normalize_telephone(text)
        
        # DIGIT - read digit by digit
        if semiotic_class == 'DIGIT':
            return cls._normalize_digit(text)
        
        # LETTERS - spell out letters
        if semiotic_class == 'LETTERS':
            return cls._normalize_letters(text)
        
        # ELECTRONIC - URLs, emails
        if semiotic_class == 'ELECTRONIC':
            return cls._normalize_electronic(text)
        
        # FRACTION - fractions
        if semiotic_class == 'FRACTION':
            return cls._normalize_fraction(text)
        
        # Default: return lowercase
        return text.lower()
    
    @classmethod
    def _normalize_punct(cls, text):
        \"\"\"Normalize punctuation to spoken form.\"\"\"
        punct_map = {{
            '.': '', ',': '', '!': '', '?': '', ';': '', ':': '',
            '-': '', '(': '', ')': '', '[': '', ']': '', '{{': '', '}}': '',
            '"': '', "'": '', '/': 'slash', '\\\\': 'backslash',
            '@': 'at', '#': 'hashtag', '$': 'dollar', '%': 'percent',
            '&': 'and', '*': 'asterisk', '+': 'plus', '=': 'equals',
            '<': 'less than', '>': 'greater than', '|': 'pipe',
            '~': 'tilde', '^': 'caret', '_': 'underscore'
        }}
        result = punct_map.get(text, '')
        return result if result else text.lower()
    
    @classmethod
    def _num_to_words(cls, n, ordinal=False):
        \"\"\"Convert number to words.\"\"\"
        if n < 0:
            return 'minus ' + cls._num_to_words(-n, ordinal)
        
        if n == 0:
            return 'zero' if not ordinal else 'zeroth'
        
        if n < 20:
            return cls.ORDINAL_ONES[n] if ordinal else cls.ONES[n]
        
        if n < 100:
            tens, ones = divmod(n, 10)
            if ones == 0:
                return cls.ORDINAL_TENS[tens] if ordinal else cls.TENS[tens]
            else:
                base = cls.TENS[tens]
                if ordinal:
                    return base + ' ' + cls.ORDINAL_ONES[ones]
                return base + ' ' + cls.ONES[ones]
        
        if n < 1000:
            hundreds, remainder = divmod(n, 100)
            result = cls.ONES[hundreds] + ' hundred'
            if remainder:
                result += ' ' + cls._num_to_words(remainder, ordinal)
            elif ordinal:
                result += 'th'
            return result
        
        if n < 1000000:
            thousands, remainder = divmod(n, 1000)
            result = cls._num_to_words(thousands) + ' thousand'
            if remainder:
                result += ' ' + cls._num_to_words(remainder, ordinal)
            elif ordinal:
                result += 'th'
            return result
        
        if n < 1000000000:
            millions, remainder = divmod(n, 1000000)
            result = cls._num_to_words(millions) + ' million'
            if remainder:
                result += ' ' + cls._num_to_words(remainder, ordinal)
            elif ordinal:
                result += 'th'
            return result
        
        billions, remainder = divmod(n, 1000000000)
        result = cls._num_to_words(billions) + ' billion'
        if remainder:
            result += ' ' + cls._num_to_words(remainder, ordinal)
        elif ordinal:
            result += 'th'
        return result
    
    @classmethod
    def _normalize_cardinal(cls, text):
        \"\"\"Normalize cardinal numbers.\"\"\"
        text = text.replace(',', '').replace(' ', '')
        try:
            # Handle decimals
            if '.' in text:
                parts = text.split('.')
                integer_part = cls._num_to_words(int(parts[0])) if parts[0] else 'zero'
                decimal_part = ' point ' + ' '.join(cls.ONES[int(d)] if d != '0' else 'zero' for d in parts[1])
                return integer_part + decimal_part
            return cls._num_to_words(int(text))
        except (ValueError, IndexError):
            return text.lower()
    
    @classmethod
    def _normalize_ordinal(cls, text):
        \"\"\"Normalize ordinal numbers.\"\"\"
        text = text.lower().replace(',', '').replace(' ', '')
        text = re.sub(r'(st|nd|rd|th)$', '', text)
        try:
            return cls._num_to_words(int(text), ordinal=True)
        except ValueError:
            return text
    
    @classmethod
    def _normalize_date(cls, text):
        \"\"\"Normalize date expressions.\"\"\"
        text = text.strip()
        
        # Try various date formats
        # Format: MM/DD/YYYY or MM-DD-YYYY
        match = re.match(r'(\\d{{1,2}})[/-](\\d{{1,2}})[/-](\\d{{2,4}})', text)
        if match:
            month, day, year = int(match.group(1)), int(match.group(2)), match.group(3)
            month_str = cls.MONTHS.get(month, str(month))
            day_str = cls._num_to_words(day, ordinal=True)
            if len(year) == 2:
                year = '20' + year if int(year) < 50 else '19' + year
            year_str = cls._normalize_year(year)
            return f'{{month_str}} {{day_str}} {{year_str}}'
        
        # Format: YYYY
        if re.match(r'^\\d{{4}}$', text):
            return cls._normalize_year(text)
        
        return text.lower()
    
    @classmethod
    def _normalize_year(cls, year):
        \"\"\"Normalize year to spoken form.\"\"\"
        year = str(year)
        if len(year) == 4:
            y = int(year)
            if 2000 <= y <= 2009:
                return 'two thousand ' + (cls.ONES[y - 2000] if y > 2000 else '')
            elif 2010 <= y <= 2099:
                return 'twenty ' + cls._num_to_words(y - 2000)
            elif 1900 <= y <= 1999:
                return 'nineteen ' + cls._num_to_words(y - 1900)
            else:
                first_half = y // 100
                second_half = y % 100
                return cls._num_to_words(first_half) + ' ' + cls._num_to_words(second_half)
        return cls._normalize_cardinal(year)
    
    @classmethod
    def _normalize_time(cls, text):
        \"\"\"Normalize time expressions.\"\"\"
        match = re.match(r'(\\d{{1,2}}):(\\d{{2}})(?::(\\d{{2}}))?\\s*(am|pm|a\\.m\\.|p\\.m\\.)?', text.lower())
        if match:
            hour, minute = int(match.group(1)), int(match.group(2))
            second = match.group(3)
            period = match.group(4)
            
            result = cls._num_to_words(hour)
            if minute == 0:
                result += " o'clock" if not period else ''
            else:
                result += ' ' + cls._num_to_words(minute)
            
            if second and int(second) > 0:
                result += ' and ' + cls._num_to_words(int(second)) + ' seconds'
            
            if period:
                period = period.replace('.', '')
                result += ' ' + ' '.join(period)
            
            return result
        return text.lower()
    
    @classmethod
    def _normalize_measure(cls, text):
        \"\"\"Normalize measurements.\"\"\"
        units = {{
            'km': 'kilometers', 'cm': 'centimeters', 'mm': 'millimeters', 'm': 'meters',
            'kg': 'kilograms', 'g': 'grams', 'mg': 'milligrams', 'lb': 'pounds', 'lbs': 'pounds',
            'oz': 'ounces', 'ml': 'milliliters', 'l': 'liters', 'ft': 'feet', 'in': 'inches',
            'mi': 'miles', 'yd': 'yards', 'mph': 'miles per hour', 'kph': 'kilometers per hour',
            '%': 'percent', '°': 'degrees', '°c': 'degrees celsius', '°f': 'degrees fahrenheit'
        }}
        
        match = re.match(r'([\\d,.]+)\\s*([a-zA-Z%°]+)', text)
        if match:
            number, unit = match.groups()
            unit_lower = unit.lower()
            unit_text = units.get(unit_lower, unit_lower)
            num_text = cls._normalize_cardinal(number)
            return f'{{num_text}} {{unit_text}}'
        return text.lower()
    
    @classmethod
    def _normalize_money(cls, text):
        \"\"\"Normalize monetary amounts.\"\"\"
        currencies = {{
            '$': 'dollars', '£': 'pounds', '€': 'euros', '¥': 'yen',
            'usd': 'dollars', 'gbp': 'pounds', 'eur': 'euros'
        }}
        
        match = re.match(r'([$£€¥]?)([\\d,.]+)\\s*(\\w*)', text)
        if match:
            symbol, amount, suffix = match.groups()
            currency = currencies.get(symbol.lower() or suffix.lower(), 'dollars')
            
            # Handle cents
            if '.' in amount:
                dollars, cents = amount.split('.')
                dollars = cls._normalize_cardinal(dollars)
                cents = cls._normalize_cardinal(cents.ljust(2, '0')[:2])
                if int(cents.replace(' ', '')) > 0:
                    return f'{{dollars}} {{currency}} and {{cents}} cents'
                return f'{{dollars}} {{currency}}'
            return f'{{cls._normalize_cardinal(amount)}} {{currency}}'
        return text.lower()
    
    @classmethod
    def _normalize_telephone(cls, text):
        \"\"\"Normalize telephone numbers - read digit by digit.\"\"\"
        digits = re.findall(r'\\d', text)
        digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        return ' '.join(digit_words[int(d)] for d in digits)
    
    @classmethod
    def _normalize_digit(cls, text):
        \"\"\"Normalize digit sequences - read digit by digit.\"\"\"
        digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        result = []
        for char in text:
            if char.isdigit():
                result.append(digit_words[int(char)])
            elif char == '.':
                result.append('point')
            elif char == '-':
                result.append('')
        return ' '.join(filter(None, result))
    
    @classmethod
    def _normalize_letters(cls, text):
        \"\"\"Normalize letter sequences - spell out.\"\"\"
        return ' '.join(char.lower() for char in text if char.isalpha())
    
    @classmethod
    def _normalize_electronic(cls, text):
        \"\"\"Normalize electronic addresses (URLs, emails).\"\"\"
        text = text.lower()
        replacements = {{
            'http://': 'h t t p colon slash slash ',
            'https://': 'h t t p s colon slash slash ',
            'www.': 'w w w dot ',
            '.com': ' dot com',
            '.org': ' dot org',
            '.net': ' dot net',
            '.edu': ' dot edu',
            '.gov': ' dot gov',
            '@': ' at ',
            '/': ' slash ',
            '.': ' dot ',
            '-': ' dash ',
            '_': ' underscore '
        }}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return ' '.join(text.split())
    
    @classmethod
    def _normalize_fraction(cls, text):
        \"\"\"Normalize fractions.\"\"\"
        common_fractions = {{
            '1/2': 'one half', '1/3': 'one third', '2/3': 'two thirds',
            '1/4': 'one quarter', '3/4': 'three quarters',
            '1/5': 'one fifth', '2/5': 'two fifths', '3/5': 'three fifths', '4/5': 'four fifths',
            '1/8': 'one eighth', '3/8': 'three eighths', '5/8': 'five eighths', '7/8': 'seven eighths'
        }}
        if text in common_fractions:
            return common_fractions[text]
        
        match = re.match(r'(\\d+)/(\\d+)', text)
        if match:
            num, denom = int(match.group(1)), int(match.group(2))
            num_word = cls._num_to_words(num)
            denom_word = cls._num_to_words(denom, ordinal=True)
            if num > 1:
                denom_word += 's'
            return f'{{num_word}} {{denom_word}}'
        return text.lower()


# ============================================================================
# MAIN PROCESSING LOGIC
# ============================================================================

print("Loading data...")
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

print(f"Train shape: {{train_df.shape}}")
print(f"Test shape: {{test_df.shape}}")

# Detect column names
before_col = None
after_col = None
class_col = None
id_col = None
sentence_id_col = None

for col in train_df.columns:
    col_lower = col.lower()
    if 'before' in col_lower:
        before_col = col
    elif 'after' in col_lower:
        after_col = col
    elif 'class' in col_lower:
        class_col = col
    elif 'sentence' in col_lower and 'id' in col_lower:
        sentence_id_col = col
    elif col_lower in ['id', 'row_id', 'index']:
        id_col = col

# Fallback column detection
if before_col is None:
    before_col = train_df.columns[1] if len(train_df.columns) > 1 else train_df.columns[0]
if after_col is None:
    after_col = train_df.columns[2] if len(train_df.columns) > 2 else before_col
if class_col is None:
    for col in train_df.columns:
        if train_df[col].dtype == 'object' and train_df[col].nunique() < 50:
            class_col = col
            break

print(f"Detected columns - before: {{before_col}}, after: {{after_col}}, class: {{class_col}}")

# Apply rule-based normalization
print("Applying rule-based normalization to test data...")
predictions = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Normalizing"):
    before_text = str(row[before_col]) if before_col and before_col in test_df.columns else ''
    
    # Get class if available, else try to infer
    if class_col and class_col in test_df.columns:
        semiotic_class = str(row[class_col])
    else:
        # Simple class inference based on content
        if before_text.replace(',', '').replace('.', '').replace('-', '').isdigit():
            semiotic_class = 'CARDINAL'
        elif re.match(r'^\\d{{1,2}}[/-]\\d{{1,2}}[/-]\\d{{2,4}}$', before_text):
            semiotic_class = 'DATE'
        elif re.match(r'^\\d{{1,2}}:\\d{{2}}', before_text):
            semiotic_class = 'TIME'
        elif re.match(r'^[$£€]', before_text):
            semiotic_class = 'MONEY'
        elif len(before_text) == 1 and not before_text.isalnum():
            semiotic_class = 'PUNCT'
        else:
            semiotic_class = 'PLAIN'
    
    normalized = RuleBasedNormalizer.normalize(before_text, semiotic_class)
    predictions.append(normalized)

# Detect ID column for test data
test_id_col = None
for col in test_df.columns:
    col_lower = col.lower()
    if col_lower in ['id', 'row_id', 'index']:
        test_id_col = col
        break

if test_id_col is None:
    test_id_col = test_df.columns[0]

# Create submission
print("Creating submission file...")
submission = pd.DataFrame({{
    'id': test_df[test_id_col] if test_id_col in test_df.columns else range(len(test_df)),
    '{prediction_column}': predictions
}})

# Handle multi-column submission if needed
if sentence_id_col and sentence_id_col in test_df.columns:
    submission['sentence_id'] = test_df[sentence_id_col]
    submission['token_id'] = test_df.index

submission.to_csv('submission.csv', index=False)
print(f"Submission saved with {{len(submission)}} rows")
print("Text normalization complete!")
"""

SEQ2SEQ_T5_TEMPLATE = """
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class Seq2SeqDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_source_length=128, max_target_length=128):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        source = str(self.sources[idx])
        target = str(self.targets[idx]) if self.targets is not None else ""
        
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if self.targets is not None:
            target_encoding = self.tokenizer(
                target,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels = target_encoding['input_ids'].flatten()
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = None
        
        return {{
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': labels
        }}

# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

# Prepare data
X_train = train_df['{source_column}'].values
y_train = train_df['{target_column}'].values
X_test = test_df['{source_column}'].values

# Split for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.1, random_state={seed}
)

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model = model.to(device)

# Create datasets
train_dataset = Seq2SeqDataset(
    X_train_split, y_train_split, tokenizer,
    max_source_length={max_source_length},
    max_target_length={max_target_length}
)
val_dataset = Seq2SeqDataset(
    X_val_split, y_val_split, tokenizer,
    max_source_length={max_source_length},
    max_target_length={max_target_length}
)
test_dataset = Seq2SeqDataset(
    X_test, None, tokenizer,
    max_source_length={max_source_length},
    max_target_length={max_target_length}
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size={batch_size}, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr={learning_rate}, weight_decay={weight_decay})

# Training loop
best_val_loss = float('inf')
patience_counter = 0

for epoch in range({max_epochs}):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch {{epoch+1}}: Train Loss = {{avg_train_loss:.4f}}, Val Loss = {{avg_val_loss:.4f}}')
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= {early_stopping_patience}:
            print(f'Early stopping at epoch {{epoch+1}}')
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Inference with beam search
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Inference'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length={max_target_length},
            num_beams={beam_search_size},
            length_penalty={length_penalty},
            early_stopping=True
        )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

# Save predictions
submission = pd.DataFrame({{
    '{id_column}': test_df['{id_column}'],
    '{prediction_column}': predictions
}})
submission.to_csv('submission.csv', index=False)

print("Training and inference complete")
"""

SEQ2SEQ_RESOURCE_CONSTRAINED_TEMPLATE = """
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class Seq2SeqDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_source_length=64, max_target_length=64):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        source = str(self.sources[idx])
        target = str(self.targets[idx]) if self.targets is not None else ""
        
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if self.targets is not None:
            target_encoding = self.tokenizer(
                target,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels = target_encoding['input_ids'].flatten()
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = None
        
        return {{
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': labels
        }}

# Load data
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

# Prepare data
X_train = train_df['{source_column}'].values
y_train = train_df['{target_column}'].values
X_test = test_df['{source_column}'].values

# Split for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.1, random_state={seed}
)

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model = model.to(device)

# Create datasets with reduced lengths
train_dataset = Seq2SeqDataset(
    X_train_split, y_train_split, tokenizer,
    max_source_length=64,
    max_target_length=64
)
val_dataset = Seq2SeqDataset(
    X_val_split, y_val_split, tokenizer,
    max_source_length=64,
    max_target_length=64
)
test_dataset = Seq2SeqDataset(
    X_test, None, tokenizer,
    max_source_length=64,
    max_target_length=64
)

# Create dataloaders with smaller batch size
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size={batch_size}, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr={learning_rate}, weight_decay={weight_decay})

# Training loop with gradient accumulation
best_val_loss = float('inf')
patience_counter = 0
accumulation_steps = {gradient_accumulation_steps}

for epoch in range({max_epochs}):
    # Training
    model.train()
    train_loss = 0.0
    
    for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {{epoch+1}}/{{max_epochs}}')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accumulation_steps
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch {{epoch+1}}: Train Loss = {{avg_train_loss:.4f}}, Val Loss = {{avg_val_loss:.4f}}')
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= {early_stopping_patience}:
            print(f'Early stopping at epoch {{epoch+1}}')
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Inference with reduced beam search
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Inference'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            num_beams=2,
            length_penalty=0.6,
            early_stopping=True
        )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

# Save predictions
submission = pd.DataFrame({{
    '{id_column}': test_df['{id_column}'],
    '{prediction_column}': predictions
}})
submission.to_csv('submission.csv', index=False)

print("Training and inference complete")
"""


def get_seq2seq_template(resource_constrained: bool = False, is_text_normalization: bool = False) -> str:
    """
    Get appropriate seq2seq template based on resource constraints and task type.
    
    Args:
        resource_constrained: Whether to use resource-constrained variant
        is_text_normalization: Whether this is a text normalization task
        
    Returns:
        Template string
    """
    if is_text_normalization:
        return TEXT_NORMALIZATION_TEMPLATE
    elif resource_constrained:
        return SEQ2SEQ_RESOURCE_CONSTRAINED_TEMPLATE
    else:
        return SEQ2SEQ_T5_TEMPLATE
