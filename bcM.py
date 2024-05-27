import hashlib
import datetime
import json
import os
from urllib.parse import urlparse
import requests

class Block:
    def __init__(self, index, previous_hash, timestamp, transactions, proof, hash=None):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.proof = proof
        self.hash = hash or self.hash_block()

    def hash_block(self):
        sha = hashlib.sha256()
        sha.update(str(self.index).encode('utf-8') +
                    str(self.previous_hash).encode('utf-8') +
                    str(self.timestamp).encode('utf-8') +
                    str(self.transactions).encode('utf-8') +
                    str(self.proof).encode('utf-8'))
        return sha.hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.nodes = set()
        # Create genesis block
        self.new_block(previous_hash='1', proof=100)

    def new_block(self, proof, previous_hash=None):
        block = Block(
            index=len(self.chain) + 1,
            timestamp=datetime.datetime.now(),
            transactions=self.current_transactions,
            proof=proof,
            previous_hash=previous_hash or self.hash(self.chain[-1]),
        )
        self.current_transactions = []
        self.chain.append(block)
        self.save_blockchain() 
        return block

    def new_transaction(self, doc, pat, resss):
        self.current_transactions.append({
            'Doctor': doc,
            'Patient': pat,
            'Result': resss,
        })
        return self.last_block.index + 1

    @staticmethod
    def hash(block):
        return block.hash_block()

    @property
    def last_block(self):
        return self.chain[-1]

    def proof_of_work(self, last_proof):
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof):
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    def register_node(self, address):
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)

    def valid_chain(self, chain):
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            if block.previous_hash != self.hash(last_block):
                return False
            if not self.valid_proof(last_block.proof, block.proof):
                return False
            last_block = block
            current_index += 1

        return True

    def resolve_conflicts(self):
        neighbors = self.nodes
        new_chain = None
        max_length = len(self.chain)

        for node in neighbors:
            response = requests.get(f'http://{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        if new_chain:
            self.chain = new_chain
            self.save_blockchain()
            return True

        return False

    def save_blockchain(self):
        with open(r'D:\Project\MpoxDetect\blockchain.json', 'w') as file:
            chain_data = []
            for block in self.chain:
                chain_data.append({
                    'index': block.index,
                    'timestamp': str(block.timestamp),
                    'transactions': block.transactions,
                    'proof': block.proof,
                    'previous_hash': block.previous_hash,
                    'hash': block.hash
                })
            json.dump(chain_data, file, indent=4)

def load_blockchain():
    if os.path.exists('blockchain.json'):
        with open('blockchain.json', 'r') as file:
            chain_data = json.load(file)
            blockchain = Blockchain()
            for block_data in chain_data:
                block = Block(
                    index=block_data['index'],
                    previous_hash=block_data['previous_hash'],
                    timestamp=datetime.datetime.strptime(block_data['timestamp'], "%Y-%m-%d %H:%M:%S.%f"),
                    transactions=block_data['transactions'],
                    proof=block_data['proof'],
                    hash=block_data['hash']
                )
                blockchain.chain.append(block)
            return blockchain
    else:
        return Blockchain()

def bcm(Pname,Dname,Ress):
    blockchain = load_blockchain()
    blockchain.new_transaction(Dname,Pname,Ress)
    last_block = blockchain.last_block
    last_proof = last_block.proof
    proof = blockchain.proof_of_work(last_proof)
    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(proof, previous_hash)
    blockchain.save_blockchain()  

if __name__ == "__main__":
    main()
