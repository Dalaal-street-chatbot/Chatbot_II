import React from 'react';
import ChatWindow from './ChatWindow';
import './DualInterface.css';

interface DualInterfaceProps {}

const DualInterface: React.FC<DualInterfaceProps> = () => (
  <div className="dual-interface">
    <h1>💰 Dalaal Street Chatbot</h1>
    <ChatWindow />
  </div>
);

export default DualInterface;
