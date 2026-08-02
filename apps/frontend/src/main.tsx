import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'

// Order matters: the design system first, then the token overrides, then the
// app layer that consumes both. Tailwind's index.css stays until the last
// component stops using its utilities.
import './styles/design-system.css'
import './styles/theme.css'
import './styles/app.css'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
