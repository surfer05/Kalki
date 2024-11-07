# main.py

"""
Main simulation script for the Key Lattice Group Messaging Protocol.
"""

import asyncio
import os
import hashlib

from participant import Participant
from utils import KeyRoll, increment

# Constants
NUM_PARTICIPANTS = 3  # Number of participants
WINDOW_SIZE = 3       # Window size for forgetting keys
NUM_CYCLES = 4        # Number of cycles to run the simulation

async def simulate_protocol(participant_objects, participants):
    """
    Simulate the group messaging protocol over a number of cycles.
    """
    for cycle in range(1, NUM_CYCLES + 1):
        print(f"\n=== Cycle {cycle} ===")
        # Each participant may perform a key update
        await asyncio.gather(
            *[participant_objects[i].maybe_send_update(cycle) for i in participants]
        )
        # Each participant sends a message
        await asyncio.gather(
            *[participant_objects[i].send_message(f"Message from participant {i} at cycle {cycle}", cycle) for i in participants]
        )

        # Visualize the lattice for each participant at the end of each cycle
        for i in participants:
            participant_objects[i].lattice.visualize_lattice(i, cycle)

        # After the window size is reached, participants forget old keys
        if cycle >= WINDOW_SIZE:
            for p in participant_objects.values():
                with p.lattice.lattice_lock:
                    max_index = p.lattice.max_index
                    window_index = tuple(max(i - WINDOW_SIZE + 1, 0) for i in max_index)
                print(f"Participant {p.index} is forgetting keys with indices <= {window_index}")
                p.lattice.forget(window_index)
                # Visualize the lattice after forgetting keys
                p.lattice.visualize_lattice(p.index, f"{cycle} (after forgetting)")

        print()
    # Allow time for any remaining buffered messages to be decrypted
    await asyncio.sleep(1)
    # Optionally, after the simulation, print the final lattice state for each participant
    for i in participants:
        print(f"\nFinal lattice state for Participant {i}:")
        participant_objects[i].lattice.print_lattice()

if __name__ == "__main__":
    # Participants list
    participants = list(range(NUM_PARTICIPANTS))

    # Generate pairwise keys
    pairwise_keys = {}  # Key: (i, j), Value: shared key
    for i in participants:
        for j in participants:
            if i < j:
                key = os.urandom(32)
                pairwise_keys[(i, j)] = key

    # Initialize participants with initial key k0
    initial_key = os.urandom(32)

    # Print the initial key k0 (represented securely)
    initial_key_repr = hashlib.sha256(initial_key).hexdigest()[:8]
    print(f"Initial group key k0 hash: {initial_key_repr}\n")

    # Initialize participants with their pairwise keys
    participant_objects = {i: None for i in participants}
    for i in participants:
        participant_pairwise_keys = {}
        for j in participants:
            if i != j:
                if i < j:
                    key = pairwise_keys[(i, j)]
                else:
                    key = pairwise_keys[(j, i)]
                participant_pairwise_keys[j] = key
        participant_objects[i] = Participant(
            index=i,
            initial_key=initial_key,
            participant_pairwise_keys=participant_pairwise_keys,
            dimension=NUM_PARTICIPANTS,
            participants=participants,
            participant_objects=participant_objects  # Pass the dictionary itself
        )

    # Run the simulation
    asyncio.run(simulate_protocol(participant_objects, participants))
