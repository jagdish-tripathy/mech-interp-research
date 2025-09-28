'''

Version 2 creates too many rejection patterns and over-corrects for under-representation of rejection tokens in V1.

📊 Pattern Statistics:
   Rejection patterns: 410
   Pending patterns: 52
   Approval stems: 42
✅ Token verification complete:
   Approval tokens: 757
   Rejection tokens: 3769
   Pending tokens: 864

'''

import os
import numpy as np
from transformers import AutoTokenizer

# Assuming you have DATA_PATH defined
# DATA_PATH = "your/data/path"

def load_gemma_tokenizer():
    """Load Gemma-7B tokenizer"""
    print("🔄 Loading Gemma-7B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    print(f"✅ Gemma-7B tokenizer loaded. Vocab size: {len(tokenizer.vocab):,}")
    return tokenizer

# Check if we already have verified tokens saved
token_cache_file = os.path.join(DATA_PATH, 'gemma_verified_tokens.npz')

# Load tokenizer
tokenizer = load_gemma_tokenizer()

# CORRECTED: Always try to load from cache first, then verify if needed
if os.path.exists(token_cache_file):
    print(f"📁 Found cached tokens: {token_cache_file}")

    # Load from cache file
    try:
        cached_data = np.load(token_cache_file)
        approval_token_ids = cached_data['approval_ids'].tolist()
        rejection_token_ids = cached_data['rejection_ids'].tolist()
        pending_token_ids = cached_data['pending_ids'].tolist()

        print("✅ Tokens loaded from cache:")
        print(f"   Approval tokens: {len(approval_token_ids)}")
        print(f"   Rejection tokens: {len(rejection_token_ids)}")
        print(f"   Pending tokens: {len(pending_token_ids)}")

    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        print("🔄 Will regenerate tokens...")
        os.remove(token_cache_file)  # Delete corrupted cache
        approval_token_ids = None  # Force regeneration

else:
    print("🔍 No cached tokens found. Generating new tokens...")
    approval_token_ids = None  # Force generation

# Generate tokens if not loaded from cache
if 'approval_token_ids' not in locals() or approval_token_ids is None:
    print("🔍 Verifying Gemma-7B vocabulary tokens...")

    def find_approval_rejection_tokens(tokenizer):
        """Find ALL tokens in Gemma-7B vocabulary that relate to approval/rejection"""
        vocab = tokenizer.vocab
        vocab_size = len(vocab)
        print(f"Scanning {vocab_size:,} tokens in Gemma-7B vocabulary...")

        approval_tokens = []
        rejection_tokens = []
        pending_tokens = []

        # Create reverse mapping from token_id to token_string
        id_to_token = {v: k for k, v in vocab.items()}

        # PRE-COMPUTE ALL PATTERN LISTS ONCE (outside the loop)
        # REJECTION PATTERNS
        # REJECTION PATTERNS
        rejection_base_patterns = [
            # Direct rejection terms  
            'reject', 'decline', 'deny', 'refuse', 'disapprov', 'dismiss',
            
            # Disqualification terms
            'disqualified', 'ineligible', 'inadequate', 'insufficient', 
            'unacceptable', 'unsatisfactory', 'inappropriate', 'unsuitable',
            
            # Negative outcomes
            'fail', 'unsuccessful', 'unfavorable', 'adverse', 'poor', 'weak', 'deficient',
            'bad', 'terrible', 'awful', 'horrible', 'wrong', 'incorrect', 'false',
            
            # Risk and concern terms
            'risky', 'unsafe', 'dangerous', 'problematic', 'concerning', 'warning', 'alert',
            'suspicious', 'questionable', 'doubtful', 'uncertain', 'unreliable',
            
            # Financial rejection terms
            'default', 'bankrupt', 'insolvent', 'delinquent', 'overdue', 'foreclos',
            'subprime', 'toxic', 'junk', 'distressed', 'troubled', 'failing',
            
            # Capability/quality negatives
            'incompetent', 'incapable', 'unable', 'lacking', 'missing', 'absent',
            'limited', 'restricted', 'constrained', 'impaired', 'compromised',
            
            # Explicit negatives and stoppers
            'impossible', 'forbidden', 'prohibited', 'banned', 'blocked', 'terminated', 
            'cancelled', 'stopped', 'halted', 'suspended', 'withdrawn', 'revoked',
            
            # Denial and refusal variations
            'no', 'not', 'never', 'none', 'nothing', 'nobody', 'nowhere',
            'cannot', "can't", 'wont', "won't", 'shouldnt', "shouldn't",
            
            # Problem indicators
            'issue', 'problem', 'trouble', 'difficulty', 'challenge', 'obstacle',
            'barrier', 'impediment', 'hurdle', 'complication', 'error', 'mistake',
            
            # Severity indicators
            'severe', 'serious', 'major', 'critical', 'significant', 'substantial',
            'extreme', 'excessive', 'overwhelming', 'unmanageable'
        ]
        
        # Add negation variants for positive terms (EXPANDED)
        negation_prefixes = ['un', 'in', 'non', 'dis', 'im', 'ir', 'il', 'mis', 'mal', 'anti']
        positive_stems = [
            'qualified', 'suitable', 'acceptable', 'favorable', 'viable', 'approved', 
            'adequate', 'sufficient', 'satisfactory', 'appropriate', 'worthy', 'deserving',
            'reliable', 'trustworthy', 'stable', 'secure', 'sound', 'healthy', 'strong',
            'competitive', 'attractive', 'appealing', 'desirable', 'optimal', 'ideal',
            'profitable', 'beneficial', 'advantageous', 'valuable', 'effective'
        ]
        
        rejection_patterns = rejection_base_patterns.copy()
        for prefix in negation_prefixes:
            for stem in positive_stems:
                rejection_patterns.append(f'{prefix}{stem}')  # e.g., "unqualified", "inviable", "inadequate"
        
        # Special handling for negative words that shouldn't catch positives
        rejection_exact_matches = ['no', 'bad', 'never', 'cannot', "can't", 'negative', 'void']

        # PENDING PATTERNS
        pending_patterns = [
            # Review processes
            'pending', 'review', 'reviewing', 'under', 'consideration', 'consider',
            'evaluat', 'assess', 'analyzing', 'processing', 'investigat',
            
            # Time-related pending
            'wait', 'waiting', 'hold', 'holding', 'delay', 'defer', 'postpone',
            'suspend', 'pause', 'interim', 'temporary', 'provisional',
            
            # Information gathering
            'check', 'checking', 'verify', 'verifying', 'confirm', 'confirming',
            'validate', 'research', 'investigate', 'examine', 'audit',
            
            # Decision pending
            'undecided', 'uncertain', 'unclear', 'tbd', 'determining',
            'deliberat', 'contemplat', 'weighing', 'studying',
            
            # Conditional states
            'conditional', 'tentative', 'preliminary', 'partial', 'incomplete',
            'ongoing', 'active', 'open', 'progress'
        ]

        # APPROVAL PATTERNS  
        approval_base_stems = [
            # Core approval stems
            'approv', 'accept', 'grant', 'author', 'sanction', 'endorse',
            
            # Qualification stems  
            'qualif', 'eligib', 'suitab', 'appropriat', 'adequat', 'satisfactor',
            
            # Positive outcome stems
            'confirm', 'agree', 'consent', 'ratif', 'validat', 'certif', 'pass', 'success',
            
            # Quality indicator stems
            'excellent', 'outstanding', 'superior', 'strong', 'solid', 'favorab', 'positiv',
            
            # Financial approval stems
            'creditworth', 'reliabl', 'trustworth', 'stab', 'secur', 'sound', 'viabl', 'profit',
            
            # Achievement stems
            'merit', 'earn', 'deserv', 'warrant', 'justif', 'exceed', 'meet'
        ]
        
        # Simple affirmative words (exact match only)
        approval_exact_matches = ['yes', 'ok', 'okay', 'fine', 'good', 'right', 'perfect', 'ideal']

        print(f"📊 Pattern Statistics:")
        print(f"   Rejection patterns: {len(rejection_patterns)}")
        print(f"   Pending patterns: {len(pending_patterns)}")
        print(f"   Approval stems: {len(approval_base_stems)}")

        # NOW LOOP THROUGH TOKENS (patterns are pre-computed)
        for token_id in range(vocab_size):
            if token_id not in id_to_token:
                continue
                
            token_str = id_to_token[token_id]
            # Clean token string (remove special prefixes like ▁ in SentencePiece)
            clean_token = token_str.replace('▁', '').replace('Ġ', '').strip()
            token_lower = clean_token.lower()

            # Skip very short tokens or special tokens
            if len(clean_token) < 2 or token_str.startswith('<') or token_str.startswith('['):
                continue
            
            # Check for rejection patterns
            is_rejection = False
            
            # First check exact matches (for short negative words)
            if clean_token in rejection_exact_matches:
                is_rejection = True
            # Then check substring patterns
            elif any(pattern in token_lower for pattern in rejection_patterns):
                is_rejection = True
            
            if is_rejection:
                rejection_tokens.append((token_id, token_str, clean_token))
                continue

            # Check for pending patterns
            if any(pattern in token_lower for pattern in pending_patterns):
                pending_tokens.append((token_id, token_str, clean_token))
                continue

            # Check for approval patterns
            is_approval = False
            
            # First check exact matches
            if clean_token in approval_exact_matches:
                is_approval = True
            # Then check stem patterns
            elif any(stem in token_lower for stem in approval_base_stems):
                # CRITICAL: Exclude negated versions that should be rejections
                negated_indicators = ['un', 'in', 'dis', 'non', 'im', 'not', 'never']
                if any(neg in token_lower for neg in negated_indicators):
                    is_approval = False  # This should be caught by rejection patterns instead
                else:
                    is_approval = True
            
            if is_approval:
                approval_tokens.append((token_id, token_str, clean_token))
                continue

        return approval_tokens, rejection_tokens, pending_tokens

    # Find and cache tokens
    approval_tokens, rejection_tokens, pending_tokens = find_approval_rejection_tokens(tokenizer)

    # Extract token IDs for fast lookup
    approval_token_ids = [token_id for token_id, _, _ in approval_tokens]
    rejection_token_ids = [token_id for token_id, _, _ in rejection_tokens]
    pending_token_ids = [token_id for token_id, _, _ in pending_tokens]

    # Save for future runs - FIXED the typo in original code
    np.savez(token_cache_file,
             approval_ids=approval_token_ids,
             rejection_ids=rejection_token_ids,
             pending_ids=pending_token_ids)

    print(f"✅ Token verification complete:")
    print(f"   Approval tokens: {len(approval_token_ids)}")
    print(f"   Rejection tokens: {len(rejection_token_ids)}")
    print(f"   Pending tokens: {len(pending_token_ids)}")
    print(f"   💾 Saved to: {token_cache_file}")

    # Display sample tokens for verification
    print(f"\n📋 SAMPLE TOKENS FOUND:")
    print(f"Approval samples: {[clean_token for _, _, clean_token in approval_tokens[:10]]}")
    print(f"Rejection samples: {[clean_token for _, _, clean_token in rejection_tokens[:10]]}")
    print(f"Pending samples: {[clean_token for _, _, clean_token in pending_tokens[:10]]}")

# Final verification
print(f"\n🎯 READY FOR PROCESSING:")
print(f"   ✅ approval_token_ids: {type(approval_token_ids)} with {len(approval_token_ids)} tokens")
print(f"   ✅ rejection_token_ids: {type(rejection_token_ids)} with {len(rejection_token_ids)} tokens")
print(f"   ✅ pending_token_ids: {type(pending_token_ids)} with {len(pending_token_ids)} tokens")

# Optional: Save detailed token lists for manual review
detailed_cache_file = os.path.join(DATA_PATH, 'gemma_token_details.txt')
if not os.path.exists(detailed_cache_file):
    with open(detailed_cache_file, 'w') as f:
        f.write("GEMMA-7B TOKEN ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("APPROVAL TOKENS:\n")
        for token_id, token_str, clean_token in approval_tokens:
            f.write(f"{token_id}: '{token_str}' -> '{clean_token}'\n")
        
        f.write("\nREJECTION TOKENS:\n")
        for token_id, token_str, clean_token in rejection_tokens:
            f.write(f"{token_id}: '{token_str}' -> '{clean_token}'\n")
        
        f.write("\nPENDING TOKENS:\n")
        for token_id, token_str, clean_token in pending_tokens:
            f.write(f"{token_id}: '{token_str}' -> '{clean_token}'\n")
    
    print(f"📄 Detailed token list saved to: {detailed_cache_file}")
