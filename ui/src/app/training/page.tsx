'use client'

import { useState } from 'react'

export default function Training() {
  const [featuresFile, setFeaturesFile] = useState('')
  const [depositsFile, setDepositsFile] = useState('')
  const [mineral, setMineral] = useState('Copper')
  const [message, setMessage] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          features_file: featuresFile,
          deposits_file: depositsFile,
          mineral,
        }),
      })
      const data = await response.json()
      setMessage(data.result || 'Training completed')
    } catch (error) {
      setMessage('Training failed')
    }
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8">Training</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block mb-2">Features File:</label>
          <input
            type="text"
            value={featuresFile}
            onChange={(e) => setFeaturesFile(e.target.value)}
            className="border p-2 w-full"
            placeholder="Enter features file name"
            required
          />
        </div>
        <div>
          <label className="block mb-2">Deposits File:</label>
          <input
            type="text"
            value={depositsFile}
            onChange={(e) => setDepositsFile(e.target.value)}
            className="border p-2 w-full"
            placeholder="Enter deposits file name"
            required
          />
        </div>
        <div>
          <label className="block mb-2">Mineral:</label>
          <select
            value={mineral}
            onChange={(e) => setMineral(e.target.value)}
            className="border p-2"
          >
            <option value="Copper">Copper</option>
            <option value="Diamonds">Diamonds</option>
            <option value="Gold">Gold</option>
            <option value="Lead">Lead</option>
            <option value="REE (Rare Earth Elements)">REE (Rare Earth Elements)</option>
            <option value="Tin">Tin</option>
            <option value="Uranium">Uranium</option>
          </select>
        </div>
        <button type="submit" className="bg-green-500 text-white px-4 py-2 rounded">
          Start Training
        </button>
      </form>
      {message && <p className="mt-4">{message}</p>}
    </div>
  )
}