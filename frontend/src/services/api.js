const API_BASE = '/';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}health`);
  if (!res.ok) throw new Error('后端未启动');
  return res.json();
}

export async function fetchProfile() {
  const res = await fetch(`${API_BASE}profile`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error('获取档案失败');
  return res.json();
}

/**
 * 保存用户档案（创建或更新）
 * @param {Object} profileData - { height, weight, age, gender, goal }
 * @returns {Promise<Object>}
 */
export async function saveProfile(profileData) {
  const res = await fetch(`${API_BASE}profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profileData),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '保存档案失败' }));
    throw new Error(err.detail || '保存档案失败');
  }
  return res.json();
}

export async function fetchDailyStats() {
  const res = await fetch(`${API_BASE}daily_stats`);
  if (!res.ok) throw new Error('获取统计失败');
  return res.json();
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}history`);
  if (!res.ok) throw new Error('获取历史失败');
  return res.json();
}

export async function clearHistory() {
  const res = await fetch(`${API_BASE}history`, { method: 'DELETE' });
  if (!res.ok) throw new Error('清除历史失败');
  return res.json();
}

/**
 * 上传图片文件，返回 image_url
 * @param {File} file
 * @returns {Promise<string>} imageUrl
 */
export async function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}upload-image`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '上传失败' }));
    throw new Error(err.detail || '上传失败');
  }

  const data = await res.json();
  // 返回 data_url（base64格式），可直接被 LLM 多模态模型使用
  return data.data_url;
}

/**
 * 发送聊天消息并返回 SSE 流
 * @param {string} message
 * @param {string|null} imageUrl
 * @returns {ReadableStream}
 */
export function createChatStream(message, imageUrl = null) {
  const body = { message };
  if (imageUrl) body.image_url = imageUrl;

  const response = fetch(`${API_BASE}chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  return {
    async *start() {
      const res = await response;
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: '请求失败' }));
        throw new Error(err.detail || '请求失败');
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            yield data;
          }
        }
      }
    },
    response: response,
  };
}

/**
 * 解析 SSE 数据行
 * @param {string} data
 */
export function parseSSEData(data) {
  if (data === '[DONE]') return { type: 'done' };
  try {
    const parsed = JSON.parse(data);
    if (parsed.intent) {
      return { type: 'intent', intent: parsed.intent };
    }
    return { type: 'token', value: data };
  } catch {
    return { type: 'token', value: data };
  }
}
