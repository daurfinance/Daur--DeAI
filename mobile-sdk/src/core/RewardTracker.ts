/**
 * Reward Tracker
 * Tracks and manages node rewards
 */

import { RewardInfo, RewardTransaction } from '../types';
import AsyncStorage from '@react-native-async-storage/async-storage';

export class RewardTracker {
  private rewardInfo: RewardInfo = {
    totalEarned: 0,
    available: 0,
    pending: 0,
    withdrawn: 0,
    history: [],
  };

  async initialize(): Promise<void> {
    const stored = await AsyncStorage.getItem('daur_rewards');
    if (stored) {
      this.rewardInfo = JSON.parse(stored);
    }
  }

  async getRewardInfo(): Promise<RewardInfo> {
    return { ...this.rewardInfo };
  }

  async addReward(amount: number, taskId: string, txHash?: string): Promise<void> {
    const transaction: RewardTransaction = {
      id: `reward_${Date.now()}`,
      type: 'earned',
      amount,
      timestamp: Date.now(),
      taskId,
      txHash,
    };

    this.rewardInfo.totalEarned += amount;
    this.rewardInfo.available += amount;
    this.rewardInfo.history.unshift(transaction);

    await this.save();
  }

  async recordWithdrawal(amount: number, txHash: string): Promise<void> {
    const transaction: RewardTransaction = {
      id: `withdrawal_${Date.now()}`,
      type: 'withdrawn',
      amount,
      timestamp: Date.now(),
      txHash,
    };

    this.rewardInfo.available -= amount;
    this.rewardInfo.withdrawn += amount;
    this.rewardInfo.history.unshift(transaction);

    await this.save();
  }

  private async save(): Promise<void> {
    await AsyncStorage.setItem('daur_rewards', JSON.stringify(this.rewardInfo));
  }
}
