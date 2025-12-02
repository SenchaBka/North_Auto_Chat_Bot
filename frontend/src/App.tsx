import React from 'react';
import ChatComponent from './components/ChatComponent';
import './App.css';

const App: React.FC = () => {
  return (
    <div className="App">
      <main>
        <ChatComponent />
      </main>
    </div>
  );
};

export default App;
