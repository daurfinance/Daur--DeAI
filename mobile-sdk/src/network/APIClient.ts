/**
 * API Client
 * HTTP client for DAUR backend API
 */

import { NodeInfo, TaskResult, RewardInfo } from '../types';

export class APIClient {
  constructor(private baseURL: string, private apiKey?: string) {}

  async registerNode(nodeInfo: NodeInfo): Promise<{ nodeId: string; token: string }> {
    const response = await this.request('/nodes/register', {
      method: 'POST',
      body: JSON.stringify(nodeInfo),
    });
    return response;
  }

  async submitTaskResult(result: TaskResult): Promise<void> {
    await this.request('/tasks/submit', {
      method: 'POST',
      body: JSON.stringify(result),
    });
  }

  async getRewards(nodeId: string): Promise<RewardInfo> {
    return await this.request(`/rewards/${nodeId}`);
  }

  async withdrawRewards(nodeId: string, amount: number, address: string): Promise<{ txHash: string }> {
    return await this.request('/rewards/withdraw', {
      method: 'POST',
      body: JSON.stringify({ nodeId, amount, address }),
    });
  }

  private async request(endpoint: string, options: RequestInit = {}): Promise<any> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return await response.json();
  }
}
