import Link from 'next/link'


export default function Home() {
  return (

    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8">AI Deposit Prediction System</h1>
      <p className="mb-8">Welcome to the AI Deposit Prediction System. Choose a section below:</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Link href="/upload" className="bg-blue-500 text-white p-6 rounded hover:bg-blue-600 text-center text-xl">
          📤 Data Upload
        </Link>
        <Link href="/prediction" className="bg-purple-500 text-white p-6 rounded hover:bg-purple-600 text-center text-xl">
          🔮 Run Prediction
        </Link>
        <Link href="/map" className="bg-red-500 text-white p-6 rounded hover:bg-red-600 text-center text-xl">
          🗺️ Map Visualization
        </Link>
        <Link href="/download" className="bg-green-500 text-white p-6 rounded hover:bg-green-600 text-center text-xl">
          📥 Download Results
        </Link>
      </div>
    </div>

  )
}
