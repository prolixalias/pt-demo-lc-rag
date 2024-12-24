import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import ErrorBoundary from './ErrorBoundary'
import './index.css'

console.log('Script starting...'); // Debug log

const root = document.getElementById('root');
console.log('Root element:', root); // Debug log

if (root) {
  try {
    ReactDOM.createRoot(root).render(
      <React.StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </React.StrictMode>
    );
    console.log('App rendered successfully'); // Debug log
  } catch (error) {
    console.error('Error rendering app:', error);
  }
} else {
  console.error('Root element not found');
}
