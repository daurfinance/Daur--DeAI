const TaskRegistry = artifacts.require("TaskRegistry");

module.exports = function (deployer) {
  deployer.deploy(TaskRegistry);
};
