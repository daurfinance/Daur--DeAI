// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title TaskRegistry — PoUW контракт для Daur-AI
contract TaskRegistry {
    struct Task {
        uint id;
        bytes32 taskHash;
        uint reward;
        bool verified;
        bytes32 finalHash;
    }
    struct Partial {
        uint chunkId;
        bytes32 partialHash;
        bytes zkProof;
    }
    mapping(uint => Task) public tasks;
    mapping(uint => Partial[]) public partials;
    uint public nextTaskId;

    event TaskCreated(uint indexed id, bytes32 taskHash, uint reward);
    event PartialSubmitted(uint indexed taskId, uint chunkId, bytes32 partialHash);
    event TaskVerified(uint indexed taskId, bytes32 finalHash);
    event RewardsDistributed(uint indexed taskId);

    function createTask(bytes32 taskHash, uint reward) external returns (uint) {
        uint id = nextTaskId++;
        tasks[id] = Task(id, taskHash, reward, false, 0);
        emit TaskCreated(id, taskHash, reward);
        return id;
    }

    function submitPartial(uint taskId, uint chunkId, bytes32 partialHash, bytes calldata zkProof) external {
        partials[taskId].push(Partial(chunkId, partialHash, zkProof));
        emit PartialSubmitted(taskId, chunkId, partialHash);
    }

    function aggregateAndVerify(uint taskId, bytes32 finalHash) external {
        // TODO: Вызов оракула/ИИ для проверки закономерностей
        tasks[taskId].verified = true;
        tasks[taskId].finalHash = finalHash;
        emit TaskVerified(taskId, finalHash);
    }

    function distributeRewards(uint taskId) external {
        require(tasks[taskId].verified, "Task not verified");
        // TODO: Выплата PoUW, сжигание 10%
        emit RewardsDistributed(taskId);
    }
}
