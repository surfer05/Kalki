import asyncio
import threading
# from collections import defaultdict
from aes_gcm import aes_encryption
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import cryptography.exceptions
import os
import hashlib
import random

import networkx as nx
import matplotlib.pyplot as plt


# Number of participants
n = 3
participants = [0, 1, 2]

# Initialize participants with initial key k0
initial_key = os.urandom(32)

# Generate pairwise keys
pairwise_keys = {}
for i in participants:
    for j in participants:
        if i < j:
            key = os.urandom(32)
            pairwise_keys[(i, j)] = key

def KeyRoll(k, x):
    combined = b''.join(sorted([k, x]))  # Sorting ensures commutativity
    new_key = hashlib.sha256(combined).digest()
    return new_key

def increment(index, dimension):
    new_index = list(index)
    new_index[dimension] += 1
    return tuple(new_index)

class KeyLattice:
    def __init__(self, initial_key, dimension):
        self.dimension = dimension
        self.lattice = {}  # {(index_tuple): key}
        self.lattice_lock = threading.Lock()
        initial_index = tuple([0]*self.dimension)
        self.lattice[initial_index] = initial_key
        self.max_index = initial_index

        # For visualization
        self.graph = nx.DiGraph()
        self.graph.add_node(initial_index)

    def get_max_key(self):
        return self.lattice[self.max_index], self.max_index

    def add_key(self, index, key, predecessor_index=None):
        self.lattice[index] = key
        self.update_max_index(index)

        # Update the graph for visualization
        self.graph.add_node(index)
        if predecessor_index:
            self.graph.add_edge(predecessor_index, index)

    def update_max_index(self, index):
        self.max_index = tuple(max(i, j)
        for i, j in zip(self.max_index, index))

    def print_lattice(self):
        print(f"Lattice state for dimension {self.dimension}:")
        for index in sorted(self.lattice.keys()):
            key= self.lattice[index]
            # For demonstration, we'll show the key as a hex digest of its hash
            key_repr = hashlib.sha256(key).hexdigest()[:8]
            print(f"Index {index}: Key hash {key_repr}")
        print()

    def visualize_lattice(self, participant_index):
        plt.figure(figsize=(8,6))
        pos = nx.spring_layout(self.graph, seed=42) # Position nodes for visualization
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', arrows=True)
        plt.title(f"Lattice Visualization for Participant {participant_index}")
        plt.show()

class Participant:
    def __init__(self, index, initial_key, participant_pairwise_keys):
        self.index = index
        self.lattice = KeyLattice(initial_key, n)
        self.pairwise_keys = participant_pairwise_keys
        self.buffer = []
        self.update_buffer = []
        self.received_updates = set()

    async def maybe_send_update(self, cycle):
        if random.choice([True, False]):
            await self.send_update(cycle)

    async def send_update(self, cycle):
        x = os.urandom(32)
        with self.lattice.lattice_lock:
            current_key, current_index = self.lattice.get_max_key()
            new_index = increment(current_index, self.index)
            new_key = KeyRoll(current_key, x)
            self.lattice.add_key(new_index, new_key, predecessor_index=current_index)

            # Visualization
            self.lattice.visualize_lattice(self.index)

            #  Print the key update information
            key_repr = hashlib.sha256(new_key).hexdigest()[:8]
            print(f"Cycle {cycle}: Participant {self.index} performed a key update.")
            print(f"  New index: {new_index}, Key hash: {key_repr}")
            self.lattice.print_lattice()
        update_message = {
            'sender': self.index,
            'index': new_index,
            'x': x
        }
        for participant in participants:
            if participant != self.index:
                await self.secure_send(participant, update_message)

    async def secure_send(self, recipient_index, message):
        key = self.pairwise_keys[recipient_index]
        aead = ChaCha20Poly1305(key)
        nonce = os.urandom(12)
        plaintext = str(message).encode('utf-8')
        ciphertext = aead.encrypt(nonce, plaintext, None)
        await participant_objects[recipient_index].receive_secure(self.index, nonce, ciphertext)

    async def receive_secure(self, sender_index, nonce, ciphertext):
        key = self.pairwise_keys[sender_index]
        aead = ChaCha20Poly1305(key)
        try:
            plaintext = aead.decrypt(nonce, ciphertext, None)
        except cryptography.exceptions.InvalidTag:
            print(f"Participant {self.index}: Failed to decrypt secure message from Participant {sender_index}.")
        message = eval(plaintext.decode('utf-8'))
        await self.process_update(message)

    async def process_update(self, update_message):
        sender = update_message['sender']
        index = update_message['index']
        x = update_message['x']
        if tuple(index) in self.received_updates:
            return
        self.received_updates.add(tuple(index))
        preceding_index = tuple(i - (1 if idx == sender else 0)
                                for idx, i in enumerate(index))
        with self.lattice.lattice_lock:
            if preceding_index in self.lattice.lattice:
                k = self.lattice.lattice[preceding_index]
                k_prime = KeyRoll(k, x)
                self.lattice.add_key(index, k_prime, predecessor_index=preceding_index)
                await self.process_buffered_updates()
                await self.try_decrypt_buffered_messages()
            else:
                self.update_buffer.append(update_message)
        await self.process_buffered_updates()

    async def process_buffered_updates(self):
        buffered_updates = self.update_buffer[:]
        for update_message in buffered_updates:
            sender = update_message['sender']
            index = update_message['index']
            x = update_message['x']
            preceding_index = tuple(i - (1 if idx == sender else 0)
                                    for idx, i in enumerate(index))
            with self.lattice.lattice_lock:
                if preceding_index in self.lattice.lattice:
                    k = self.lattice.lattice[preceding_index]
                    k_prime = KeyRoll(k, x)
                    self.lattice.add_key(index, k_prime)
                    self.update_buffer.remove(update_message)
        await self.try_decrypt_buffered_messages()

    async def send_message(self, message, cycle):
        with self.lattice.lattice_lock:
            max_key, max_index = self.lattice.get_max_key()
            hashed_key = hashlib.sha256(max_key).digest()
        aead = ChaCha20Poly1305(hashed_key)
        nonce = os.urandom(12)
        associated_data = str(max_index).encode('utf-8')
        ciphertext = aead.encrypt(
            nonce, message.encode('utf-8'), associated_data)
        for participant in participants:
            if participant != self.index:
                await participant_objects[participant].receive_message(nonce, ciphertext, associated_data)

    async def receive_message(self, nonce, ciphertext, associated_data):
        index = eval(associated_data.decode('utf-8'))
        with self.lattice.lattice_lock:
            if index in self.lattice.lattice:
                key = self.lattice.lattice[index]
                hashed_key = hashlib.sha256(key).digest()
                aead = ChaCha20Poly1305(hashed_key)
                try:
                    plaintext = aead.decrypt(nonce, ciphertext, associated_data)
                    print(f"Participant {self.index} received message: {plaintext.decode('utf-8')}")
                except Exception:
                    print(f"Participant {self.index} failed to decrypt the message.")
            else:
                self.buffer.append((nonce, ciphertext, associated_data))

    async def try_decrypt_buffered_messages(self):
        buffered_messages = self.buffer[:]
        for message in buffered_messages:
            nonce, ciphertext, associated_data = message
            index = eval(associated_data.decode('utf-8'))
            with self.lattice.lattice_lock:
                if index in self.lattice.lattice:
                    key = self.lattice.lattice[index]
                    hashed_key = hashlib.sha256(key).digest()
                    aead = ChaCha20Poly1305(hashed_key)
                    try:
                        plaintext = aead.decrypt(nonce, ciphertext, associated_data)
                        print(f"Participant {self.index} decrypted buffered message: {plaintext.decode('utf-8')}")
                        self.buffer.remove(message)
                    except Exception:
                        continue

    def forget(self, window_parameter):
        with self.lattice.lattice_lock:
            keys_to_delete = []
            for index in self.lattice.lattice:
                if all(i <= w for i, w in zip(index, window_parameter)):
                    keys_to_delete.append(index)
            for index in keys_to_delete:
                del self.lattice.lattice[index]


participant_objects = {}
# Initialize participants with initial key k0
initial_key = os.urandom(32)

# Print the initial key k0 (represented securely)
initial_key_repr = hashlib.sha256(initial_key).hexdigest()[:8]
print()
print(f"Initial group key k0 hash: {initial_key_repr}\n")

# Initialize participants with their pairwise keys
for i in participants:
    participant_pairwise_keys = {}
    for j in participants:
        if i != j:
            if i < j:
                key = pairwise_keys[(i, j)]
            else:
                key = pairwise_keys[(j, i)]
            participant_pairwise_keys[j] = key
    participant_objects[i] = Participant(i, initial_key, participant_pairwise_keys)

async def simulate_protocol():
    num_cycles = 4  # Number of cycles to run the simulation
    for cycle in range(1, num_cycles + 1):
        print(f"\n=== Cycle {cycle} ===")
        # Each participant may perform a key update
        await asyncio.gather(
            *[participant_objects[i].maybe_send_update(cycle) for i in participants]
        )
        # Each participant sends a message
        await asyncio.gather(
            *[participant_objects[i].send_message(f"Message from participant {i} at cycle {cycle}", cycle) for i in participants]
        )

        # Visualize the lattice for each participant
        for i in participants:
            participant_objects[i].lattice.visualize_lattice(i)

        print()
    # Allow time for any remaining buffered messages to be decrypted
    await asyncio.sleep(1)
    # Optionally, after the simulation, print the final lattice state for each participant
    for i in participants:
        print(f"\nFinal lattice state for Participant {i}:")
        participant_objects[i].lattice.print_lattice()

# Run the simulation
asyncio.run(simulate_protocol())
