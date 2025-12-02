import React, { useState, useCallback } from 'react';
import type { 
  KeyboardEvent as ReactKeyboardEvent, 
  ChangeEvent 
} from 'react';
import { sendChatQuery } from '../api/api';

interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  sources?: string[];
  toolused?: string;
}

const ChatComponent: React.FC = () => {
  const [inputQuery, setInputQuery] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const addMessage = useCallback((type: Message['type'], content: string, sources?: string[], toolused?: string) => {
    const id = Date.now().toString();
    setMessages(prev => [...prev, { id, type, content, sources, toolused }]);
  }, []);

  const handleSendQuery = useCallback(async () => {
    if (!inputQuery.trim()) return;

    const userMessage = inputQuery.trim();
    addMessage('user', userMessage);
    setInputQuery('');
    setLoading(true);

    try {
      const response = await sendChatQuery(userMessage);
      addMessage('ai', response.answer, response.sources, response.toolused);
    } catch (err) {
      addMessage('ai', `Error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  }, [inputQuery, addMessage]);

  const handleKeyDown = useCallback((e: ReactKeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendQuery();
    }
  }, [handleSendQuery]);

  const handleInputChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setInputQuery(e.target.value);
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    setInputQuery('');
  }, []);

  return (
    <div className="max-w-2xl mx-auto p-6 h-screen flex flex-col bg-gradient-to-br from-slate-50 to-indigo-50">
      <div className="flex items-center justify-between mb-6 p-6 bg-white/80 backdrop-blur-xl rounded-2xl shadow-lg border border-slate-200/50">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
          North Auto AI Chat
        </h2>
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-indigo-600 transition-all duration-200 hover:scale-105"
          >
            Clear
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6 mb-6 bg-white/50 backdrop-blur-sm rounded-2xl shadow-lg border border-slate-200/50">
        {messages.map((message) => (
          <div key={message.id} className={`flex gap-1 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`flex-1 p-4 rounded-2xl shadow-sm max-w-[80%] ${
              message.type === 'user'
                ? 'bg-indigo-500 text-white order-2'
                : 'bg-white border border-slate-200 order-1'
            }`}>
              <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
              
              {message.sources && message.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-200/50">
                  <h6 className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wide">Sources</h6>
                  <ul className="space-y-1">
                    {message.sources.map((source, idx) => (
                      <li key={idx} className="text-xs text-slate-500 break-all hover:text-indigo-600 transition-colors">
                        {source}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {message.toolused && (
                <div className="mt-2 p-2 bg-emerald-100 border border-emerald-200 rounded-lg">
                  <span className="text-xs font-mono font-semibold text-emerald-900">
                    {message.toolused}
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="flex gap-4 justify-start">
            <div className="w-8 h-8 bg-indigo-500 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            </div>
            <div className="flex-1 p-4 bg-white/50 border border-slate-200 rounded-2xl backdrop-blur-sm">
              <p className="text-slate-500 italic">AI is thinking...</p>
            </div>
          </div>
        )}
      </div>

      <div className="p-6 bg-white/80 backdrop-blur-xl rounded-2xl shadow-lg border border-slate-200/50">
        <div className="flex gap-3">
          <input
            type="text"
            value={inputQuery}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="flex-1 px-5 py-4 text-lg border-2 border-slate-200 rounded-2xl focus:border-indigo-400 focus:ring-2 focus:ring-indigo-200/50 outline-none transition-all duration-200 hover:shadow-md"
            disabled={loading}
          />
          <button
            onClick={handleSendQuery}
            disabled={loading || !inputQuery.trim()}
            className="px-8 py-4 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 disabled:opacity-50 text-white font-semibold rounded-2xl shadow-lg hover:shadow-xl hover:scale-105 active:scale-100 transition-all duration-200 disabled:cursor-not-allowed whitespace-nowrap"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatComponent;
