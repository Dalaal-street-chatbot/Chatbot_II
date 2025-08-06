import React from 'react';
import ChatWindow from './components/ChatWindow';

const App: React.FC = () => {
  return (
    <div style={{ maxWidth: 600, margin: '50px auto', border: '1px solid #ccc', borderRadius: 8, padding: 24 }}>
      <h1>Welcome to the Chat App</h1>
      <ChatWindow />
    </div>
  );
};

export default App;
