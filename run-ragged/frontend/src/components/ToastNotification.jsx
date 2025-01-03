import React, { useState, useEffect } from 'react';

const ToastNotification = ({ message, type }) => {
  const [toastVisible, setToastVisible] = useState(true);

  useEffect(() => {
    let timer;
    if (toastVisible) {
        timer = setTimeout(() => {
          setToastVisible(false);
        }, 3000); // Set the toast to disappear after 3 seconds
    }
    return () => clearTimeout(timer);
  }, [toastVisible]);

  const bgColor = type === 'success'
    ? 'bg-[#E8F5E9] border-[#006838]'
    : type === 'info'
    ? 'bg-blue-50 border-blue-400'
    : type === 'warning'
    ? 'bg-yellow-50 border-yellow-400'
    : 'bg-red-50 border-red-400';

  const textColor = type === 'success'
    ? 'text-[#006838]'
    : type === 'info'
    ? 'text-blue-700'
    : type === 'warning'
    ? 'text-yellow-700'
    : 'text-red-600';
    
  if (!toastVisible) {
      return null; // Don't render anything when the toast is not visible
    }

  return (
    <div 
      className={`
        fixed top-4 left-1/2 transform -translate-x-1/2 z-[60]
        ${bgColor} ${textColor} border rounded-lg px-4 py-2 shadow-lg
        animate-[slideDown_0.3s_ease-out]
        min-w-[200px] text-center
      `}
    >
      {message}
    </div>
  );
};

export default ToastNotification;