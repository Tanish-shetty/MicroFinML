"""
MicroFinML - Blockchain Audit Trail Module
Simulates a lightweight blockchain ledger for storing loan prediction decisions.
Demonstrates: Immutable audit transparency for microfinance credit scoring.

Each block contains:
- Loan application data
- ML model prediction (default/repay)
- Risk score
- SHA-256 hash linking to previous block
- Timestamp

This creates a tamper-proof record of all credit decisions.
"""

import hashlib
import json
import time
from datetime import datetime


class Block:
    """A single block in the audit chain."""

    def __init__(self, index, timestamp, loan_data, prediction, risk_score,
                 model_used, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.loan_data = loan_data
        self.prediction = prediction
        self.risk_score = risk_score
        self.model_used = model_used
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        """Compute SHA-256 hash of the block contents."""
        block_data = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'loan_data': self.loan_data,
            'prediction': self.prediction,
            'risk_score': self.risk_score,
            'model_used': self.model_used,
            'previous_hash': self.previous_hash
        }, sort_keys=True)
        return hashlib.sha256(block_data.encode()).hexdigest()

    def to_dict(self):
        """Convert block to dictionary."""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'loan_data': self.loan_data,
            'prediction': self.prediction,
            'risk_score': self.risk_score,
            'model_used': self.model_used,
            'previous_hash': self.previous_hash,
            'hash': self.hash
        }


class LoanAuditChain:
    """
    Blockchain-inspired audit trail for loan credit decisions.
    Provides immutable, transparent record of all ML-based loan decisions.
    """

    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the first block (genesis block) in the chain."""
        genesis = Block(
            index=0,
            timestamp=datetime.utcnow().isoformat(),
            loan_data={"description": "Genesis Block - MicroFinML Audit Chain"},
            prediction="N/A",
            risk_score=0.0,
            model_used="system",
            previous_hash="0" * 64
        )
        self.chain.append(genesis)

    def get_latest_block(self):
        """Get the most recent block in the chain."""
        return self.chain[-1]

    def add_decision(self, loan_data, prediction, risk_score, model_used):
        """
        Record a loan credit decision to the blockchain.

        Parameters:
            loan_data: dict of borrower/loan features
            prediction: str 'REPAY' or 'DEFAULT'
            risk_score: float probability of default
            model_used: str name of the ML model
        """
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.utcnow().isoformat(),
            loan_data=loan_data,
            prediction=prediction,
            risk_score=round(risk_score, 4),
            model_used=model_used,
            previous_hash=self.get_latest_block().hash
        )
        self.chain.append(new_block)
        return new_block

    def validate_chain(self):
        """
        Validate the integrity of the entire blockchain.
        Checks that no block has been tampered with.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check hash integrity
            if current.hash != current.compute_hash():
                return False, f"Block {i}: Hash mismatch (data tampered)"

            # Check chain linkage
            if current.previous_hash != previous.hash:
                return False, f"Block {i}: Chain broken (previous hash mismatch)"

        return True, "Blockchain is valid — all decisions are tamper-proof."

    def get_chain_summary(self):
        """Get a summary of the audit chain."""
        total = len(self.chain) - 1  # Exclude genesis
        if total == 0:
            return {"total_decisions": 0, "default_count": 0, "repay_count": 0}

        decisions = [b.prediction for b in self.chain[1:]]
        return {
            "total_decisions": total,
            "default_count": decisions.count("DEFAULT"),
            "repay_count": decisions.count("REPAY"),
            "default_rate": round(decisions.count("DEFAULT") / total, 4) if total > 0 else 0,
            "models_used": list(set(b.model_used for b in self.chain[1:])),
            "chain_valid": self.validate_chain()[0]
        }

    def print_chain(self, last_n=5):
        """Print the last N blocks of the chain."""
        blocks = self.chain[-last_n:] if len(self.chain) > last_n else self.chain
        print(f"\n{'='*70}")
        print(f"  BLOCKCHAIN AUDIT TRAIL (Last {len(blocks)} blocks)")
        print(f"{'='*70}")
        for block in blocks:
            print(f"\n  Block #{block.index}")
            print(f"  Timestamp:     {block.timestamp}")
            print(f"  Prediction:    {block.prediction}")
            print(f"  Risk Score:    {block.risk_score}")
            print(f"  Model:         {block.model_used}")
            print(f"  Hash:          {block.hash[:32]}...")
            print(f"  Previous Hash: {block.previous_hash[:32]}...")
            print(f"  {'─'*50}")

    def export_chain(self):
        """Export the full chain as a list of dicts."""
        return [block.to_dict() for block in self.chain]

    def to_dataframe(self):
        """Convert chain to a pandas DataFrame for analysis."""
        import pandas as pd
        records = []
        for block in self.chain[1:]:  # Skip genesis
            record = {
                'block_index': block.index,
                'timestamp': block.timestamp,
                'prediction': block.prediction,
                'risk_score': block.risk_score,
                'model_used': block.model_used,
                'hash': block.hash[:16] + '...',
                'previous_hash': block.previous_hash[:16] + '...'
            }
            # Flatten loan data
            if isinstance(block.loan_data, dict):
                for k, v in block.loan_data.items():
                    record[f'loan_{k}'] = v
            records.append(record)
        return pd.DataFrame(records)


def demo_blockchain_audit():
    """Demo: Record sample loan decisions to the blockchain."""
    chain = LoanAuditChain()

    # Sample loan decisions
    sample_decisions = [
        {
            'loan_data': {'Age': 35, 'Income': 55000, 'LoanAmount': 25000,
                          'CreditScore': 680, 'LoanPurpose': 'Home'},
            'prediction': 'REPAY', 'risk_score': 0.2834,
            'model_used': 'XGBoost'
        },
        {
            'loan_data': {'Age': 22, 'Income': 18000, 'LoanAmount': 45000,
                          'CreditScore': 420, 'LoanPurpose': 'Business'},
            'prediction': 'DEFAULT', 'risk_score': 0.8721,
            'model_used': 'XGBoost'
        },
        {
            'loan_data': {'Age': 45, 'Income': 92000, 'LoanAmount': 15000,
                          'CreditScore': 760, 'LoanPurpose': 'Education'},
            'prediction': 'REPAY', 'risk_score': 0.0923,
            'model_used': 'RandomForest'
        },
        {
            'loan_data': {'Age': 28, 'Income': 32000, 'LoanAmount': 80000,
                          'CreditScore': 380, 'LoanPurpose': 'Auto'},
            'prediction': 'DEFAULT', 'risk_score': 0.9145,
            'model_used': 'XGBoost'
        },
        {
            'loan_data': {'Age': 55, 'Income': 120000, 'LoanAmount': 50000,
                          'CreditScore': 720, 'LoanPurpose': 'Home'},
            'prediction': 'REPAY', 'risk_score': 0.1567,
            'model_used': 'LogisticRegression'
        }
    ]

    print("Recording loan decisions to blockchain audit trail...\n")
    for decision in sample_decisions:
        block = chain.add_decision(**decision)
        print(f"  Block #{block.index} added: {decision['prediction']} "
              f"(risk: {decision['risk_score']:.2%}) [{block.hash[:24]}...]")

    # Print chain
    chain.print_chain()

    # Validate
    is_valid, message = chain.validate_chain()
    print(f"\nChain Validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  {message}")

    # Summary
    summary = chain.get_chain_summary()
    print(f"\nChain Summary:")
    print(f"  Total decisions: {summary['total_decisions']}")
    print(f"  Default count:   {summary['default_count']}")
    print(f"  Repay count:     {summary['repay_count']}")
    print(f"  Default rate:    {summary['default_rate']:.2%}")
    print(f"  Models used:     {summary['models_used']}")

    # Demonstrate tamper detection
    print(f"\n{'='*70}")
    print("  TAMPER DETECTION DEMO")
    print(f"{'='*70}")
    print("\n  Attempting to tamper with Block #2 prediction...")
    original_pred = chain.chain[2].prediction
    chain.chain[2].prediction = "REPAY"  # Tamper!
    is_valid, message = chain.validate_chain()
    print(f"  Validation after tampering: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  {message}")
    chain.chain[2].prediction = original_pred  # Restore
    chain.chain[2].hash = chain.chain[2].compute_hash()  # Fix hash

    return chain


if __name__ == "__main__":
    chain = demo_blockchain_audit()
