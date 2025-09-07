"""
blockchain_client.py — Взаимодействие ядра ИИ с контрактом TaskRegistry
"""
from web3 import Web3
import json

class BlockchainClient:
    def __init__(self, rpc_url, contract_address, abi_path):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        with open(abi_path, 'r') as f:
            abi = json.load(f)
        self.contract = self.web3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)

    def create_task(self, task_hash, reward, private_key):
        account = self.web3.eth.account.from_key(private_key)
        tx = self.contract.functions.createTask(task_hash, reward).build_transaction({
            'from': account.address,
            'value': reward,
            'nonce': self.web3.eth.get_transaction_count(account.address)
        })
        signed = self.web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()

    def submit_partial(self, task_id, chunk_id, partial_hash, zk_proof, private_key):
        account = self.web3.eth.account.from_key(private_key)
        tx = self.contract.functions.submitPartial(task_id, chunk_id, partial_hash, zk_proof).build_transaction({
            'from': account.address,
            'nonce': self.web3.eth.get_transaction_count(account.address)
        })
        signed = self.web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()

    def aggregate_and_verify(self, task_id, final_hash, private_key):
        account = self.web3.eth.account.from_key(private_key)
        tx = self.contract.functions.aggregateAndVerify(task_id, final_hash).build_transaction({
            'from': account.address,
            'nonce': self.web3.eth.get_transaction_count(account.address)
        })
        signed = self.web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()

    def distribute_rewards(self, task_id, private_key):
        account = self.web3.eth.account.from_key(private_key)
        tx = self.contract.functions.distributeRewards(task_id).build_transaction({
            'from': account.address,
            'nonce': self.web3.eth.get_transaction_count(account.address)
        })
        signed = self.web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()
