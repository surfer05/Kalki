# participant.py

"""
Participant class representing a participant in the group messaging protocol.
"""

import asyncio
import os
import hashlib
import random

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import cryptography.exceptions

from key_lattice import KeyLattice
from utils import KeyRoll, increment

class Participant:
    """
    Represents a participant in the group messaging protocol.
    Manages key updates, message sending, and receiving.
    """

    def __init__(self, index, initial_key, participant_pairwise_keys, dimension, participants, participant_objects):
        self.index = index
        self.lattice = KeyLattice(initial_key, dimension)
        self.pairwise_keys = participant_pairwise_keys
        self.buffer = []
        self.update_buffer = []
        self.received_updates = set()
        self.dimension = dimension

        self.participants = participants
        self.participant_objects = participant_objects

    async def maybe_send_update(self, cycle):
        """
        Randomly decide whether to perform a key update.
        """
        if random.choice([True, False]):
            await self.send_update(cycle)

    async def send_update(self, cycle):
        """
        Perform a key update and notify other participants.
        """
        x = os.urandom(32)

        current_key, current_index = self.lattice.get_max_key()
        new_index = increment(current_index, self.index)
        new_key = KeyRoll(current_key, x)
        self.lattice.add_key(new_index, new_key, predecessor_index=current_index, x=x)
        # Print the key update information
        key_repr = hashlib.sha256(new_key).hexdigest()[:8]
        print(f"Cycle {cycle}: Participant {self.index} performed a key update.")
        print(f"  New index: {new_index}, Key hash: {key_repr}")
        update_message = {
            'sender': self.index,
            'index': new_index,
            'x': x
        }
        for participant in self.participants:
            if participant != self.index:
                await self.secure_send(participant, update_message)

    async def secure_send(self, recipient_index, message):
        """
        Securely send a message to another participant using pairwise keys.
        """
        key = self.pairwise_keys[recipient_index]
        aead = AESGCM(key)
        nonce = os.urandom(12)
        plaintext = str(message).encode('utf-8')
        ciphertext = aead.encrypt(nonce, plaintext, None)
        await self.participant_objects[recipient_index].receive_secure(self.index, nonce, ciphertext)

    async def receive_secure(self, sender_index, nonce, ciphertext):
        """
        Receive and decrypt a secure message from another participant.
        """
        key = self.pairwise_keys[sender_index]
        aead = AESGCM(key)
        try:
            plaintext = aead.decrypt(nonce, ciphertext, None)
        except cryptography.exceptions.InvalidTag:
            print(f"Participant {self.index}: Failed to decrypt secure message from Participant {sender_index}.")
            return
        message = eval(plaintext.decode('utf-8'))
        await self.process_update(message)

    async def process_update(self, update_message):
        """
        Process a key update message from another participant.
        """
        sender = update_message['sender']
        index = update_message['index']
        x = update_message['x']
        if tuple(index) in self.received_updates:
            return
        self.received_updates.add(tuple(index))
        preceding_index = tuple(i - (1 if idx == sender else 0) for idx, i in enumerate(index))
        with self.lattice.lattice_lock:
            if preceding_index in self.lattice.lattice:
                k = self.lattice.lattice[preceding_index]
                k_prime = KeyRoll(k, x)
                self.lattice.add_key(index, k_prime, predecessor_index=preceding_index, x=x)
                await self.process_buffered_updates()
                await self.try_decrypt_buffered_messages()
            else:
                self.update_buffer.append(update_message)

    async def process_buffered_updates(self):
        """
        Process any buffered key updates.
        """
        buffered_updates = self.update_buffer[:]
        for update_message in buffered_updates:
            sender = update_message['sender']
            index = update_message['index']
            x = update_message['x']
            preceding_index = tuple(i - (1 if idx == sender else 0) for idx, i in enumerate(index))
            with self.lattice.lattice_lock:
                if preceding_index in self.lattice.lattice:
                    k = self.lattice.lattice[preceding_index]
                    k_prime = KeyRoll(k, x)
                    self.lattice.add_key(index, k_prime, predecessor_index=preceding_index, x=x)
                    self.update_buffer.remove(update_message)
        await self.try_decrypt_buffered_messages()

    async def send_message(self, message, cycle):
        """
        Send a group message encrypted with the maximal key.
        """

        max_key, max_index = self.lattice.get_max_key()
        hashed_key = hashlib.sha256(max_key).digest()
        aead = AESGCM(hashed_key)
        nonce = os.urandom(12)
        associated_data = str(max_index).encode('utf-8')
        ciphertext = aead.encrypt(nonce, message.encode('utf-8'), associated_data)
        print(f"Cycle {cycle}: Participant {self.index} is sending a message with index {max_index}")
        for participant in self.participants:
            if participant != self.index:
                await self.participant_objects[participant].receive_message(nonce, ciphertext, associated_data)

    async def receive_message(self, nonce, ciphertext, associated_data):
        """
        Receive and decrypt a group message.
        """
        index = eval(associated_data.decode('utf-8'))

        if index in self.lattice.lattice:
            key = self.lattice.lattice[index]
            hashed_key = hashlib.sha256(key).digest()
            aead = AESGCM(hashed_key)
            try:
                plaintext = aead.decrypt(nonce, ciphertext, associated_data)
                print(f"Participant {self.index} received message: {plaintext.decode('utf-8')}")
            except cryptography.exceptions.InvalidTag:
                print(f"Participant {self.index} failed to decrypt the message.")
        else:
            # Buffer the message
            self.buffer.append((nonce, ciphertext, associated_data))
            print(f"Participant {self.index} buffered a message for index {index}")

    async def try_decrypt_buffered_messages(self):
        """
        Attempt to decrypt any buffered messages.
        """
        buffered_messages = self.buffer[:]
        for message in buffered_messages:
            nonce, ciphertext, associated_data = message
            index = eval(associated_data.decode('utf-8'))
            with self.lattice.lattice_lock:
                if index in self.lattice.lattice:
                    key = self.lattice.lattice[index]
                    hashed_key = hashlib.sha256(key).digest()
                    aead = AESGCM(hashed_key)
                    try:
                        plaintext = aead.decrypt(nonce, ciphertext, associated_data)
                        print(f"Participant {self.index} decrypted buffered message: {plaintext.decode('utf-8')}")
                        self.buffer.remove(message)
                    except cryptography.exceptions.InvalidTag:
                        continue
