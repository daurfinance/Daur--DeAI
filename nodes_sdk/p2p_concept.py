"""
p2p_concept.py — P2P обмен концептами между нодами через libp2p
"""
from libp2p import new_node
from libp2p.peer.peerinfo import PeerInfo
import asyncio

class ConceptP2PNode:
    def __init__(self, listen_port: int):
        self.node = None
        self.listen_port = listen_port

    async def start(self):
        self.node = await new_node(listen_port=self.listen_port)
        print(f"P2P node started on port {self.listen_port}")

    async def send_concept(self, peer_id: str, concept: dict):
        peer = PeerInfo.from_str(peer_id)
        await self.node.connect(peer)
        await self.node.send(peer, str(concept).encode())
        print(f"Sent concept to {peer_id}")

    async def receive_concept(self):
        async for peer, data in self.node.receive():
            concept = data.decode()
            print(f"Received concept from {peer.id}: {concept}")
            # Здесь можно интегрировать с ядром ИИ

# Пример использования:
# node = ConceptP2PNode(listen_port=9000)
# asyncio.run(node.start())
# asyncio.run(node.send_concept(peer_id, concept))
# asyncio.run(node.receive_concept())
