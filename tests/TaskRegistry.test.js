const TaskRegistry = artifacts.require("TaskRegistry");

contract("TaskRegistry", accounts => {
  it("should create a task", async () => {
    const registry = await TaskRegistry.deployed();
    const taskHash = web3.utils.sha3("test-task");
    const reward = 1000;
    const tx = await registry.createTask(taskHash, reward);
    assert.equal(tx.logs[0].event, "TaskCreated");
    const id = tx.logs[0].args.id.toNumber();
    const task = await registry.tasks(id);
    assert.equal(task.taskHash, taskHash);
    assert.equal(task.reward, reward);
  });

  it("should submit a partial result", async () => {
    const registry = await TaskRegistry.deployed();
    const taskId = 0;
    const chunkId = 1;
    const partialHash = web3.utils.sha3("partial");
    const zkProof = web3.utils.hexToBytes("0x1234");
    const tx = await registry.submitPartial(taskId, chunkId, partialHash, zkProof);
    assert.equal(tx.logs[0].event, "PartialSubmitted");
  });

  it("should aggregate and verify a task", async () => {
    const registry = await TaskRegistry.deployed();
    const taskId = 0;
    const finalHash = web3.utils.sha3("final");
    const tx = await registry.aggregateAndVerify(taskId, finalHash);
    assert.equal(tx.logs[0].event, "TaskVerified");
    const task = await registry.tasks(taskId);
    assert.equal(task.verified, true);
    assert.equal(task.finalHash, finalHash);
  });

  it("should distribute rewards", async () => {
    const registry = await TaskRegistry.deployed();
    const taskId = 0;
    const tx = await registry.distributeRewards(taskId);
    assert.equal(tx.logs[0].event, "RewardsDistributed");
  });
});
