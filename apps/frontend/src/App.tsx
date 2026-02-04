import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>

      {/* Tailwind CSS Test */}
      <div className="mt-8 p-6 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold text-white mb-2">🎉 Tailwind CSS is Working!</h2>
        <p className="text-white/90">If you can see this styled card, Tailwind is successfully installed.</p>
      </div>
    </>
  )
}

export default App
