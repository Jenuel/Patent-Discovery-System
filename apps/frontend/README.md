# 🎨 Patent Discovery System - Frontend

The frontend for the Patent Discovery System, built with **React**, **TypeScript**, and **Tailwind CSS**. It provides an intuitive interface for searching patents, performing prior art analysis, and exploring technology landscapes.

---

## ✨ Features

- 🔍 **Unified Search Bar**: Search patents by keyword, patent number, or technical description.
- 📂 **Patent Viewer**: Side-by-side view for reading patent claims and full texts.
- 💡 **AI Assistant Panel**: Interactive chat interface for summarizing patent findings.
- 📊 **Landscape & Analytics**: Visualizations for technology trends and CPC code distributions (Coming soon).
- 🛠️ **Rich Filtering**: Filter search results by assignee, year, CPC, and more.

---

## 🛠️ Technology Stack

- **Framework**: [React 19](https://react.dev/)
- **Build Tool**: [Vite 5+](https://vitejs.dev/)
- **Styling**: [Tailwind CSS 4+](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **API Client**: [Axios](https://axios-http.com/)
- **State Management**: React Hooks & Context API

---

## 📁 Project Structure

```text
apps/frontend/
├── src/
│   ├── components/     # UI Components (Button, Input, Card)
│   ├── features/       # Feature-specific logic (Search, Dashboard)
│   ├── hooks/          # Custom React hooks (usePatentSearch)
│   ├── lib/            # Utility libraries & API config
│   ├── types/          # TypeScript definitions
│   └── App.tsx         # Main application entry point
├── public/             # Static assets
└── index.html          # HTML template
```

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or higher)
- [npm](https://www.npmjs.com/) or [pnpm](https://pnpm.io/)

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the root of `apps/frontend/`:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Open your browser**:
   Navigate to `http://localhost:5173`.

---

## 🧪 Development

### Linting & Formatting
The project uses **ESLint** and **TypeScript** for code quality.
```bash
npm run lint
```

### Building for Production
To create an optimized production build:
```bash
npm run build
```
The output will be in the `dist/` directory.

---

## 🤝 Contributing
For contributions, please follow the guidelines in the root [README](../../README.md).
