import type { CollectionConfig } from 'payload'

const weeks = [
  ...Array.from({ length: 10 }, (_, i) => `2025-W${42 + i}`), // Weeks 42-51 of 2025
  ...Array.from({ length: 3 }, (_, i) => `2026-W${3 + i}`), // Weeks 3-5 of 2026
]

const weekDescriptions = [
  '🔰 AI Engineer Basics',
  '🧠 ML training & Neural Networks',
  '🔌 LLM training',
  '📦 LLM Catalog & Classification',
  '🧬 Embeddings',
  '🐙 GitHub Copilot Session I',
  '🐙 GitHub Copilot Session II',
  '🔍 RAG',
  '🤖 AI Agents',
  '🔗 Model Context Protocol',
  '✍️ Prompting Techniques I',
  '📖 Exercises',
  '🧩 Prompting Techniques II',
]

const Participants: CollectionConfig = {
  slug: 'participants',
  admin: {
    useAsTitle: 'nomeCompleto',
    group: 'Corso GitHub Copilot',
  },
  fields: [
    {
      type: 'row',
      fields: [
        {
          name: 'nome',
          type: 'text',
          required: true,
        },
        {
          name: 'cognome',
          type: 'text',
          required: true,
        },
      ],
    },
    {
      name: 'nomeCompleto',
      type: 'text',
      admin: { hidden: true },
      hooks: {
        afterRead: [({ data }) => `${data?.nome || ''} ${data?.cognome || ''}`.trim()],
        beforeChange: [
          ({ siblingData }) => {
            // ensures data is not stored in DB
            delete siblingData['nomeCompleto']
          },
        ],
      },
    },
    {
      name: 'email',
      type: 'email',
      required: true,
      unique: true,
    },
    {
      name: 'gruppo',
      type: 'select',
      required: true,
      options: [
        {
          label: 'Gruppo1',
          value: 'Gruppo1',
        },
        {
          label: 'Gruppo2',
          value: 'Gruppo2',
        },
      ],
    },
    {
      name: 'presenze',
      admin: {
        description: ' ',
      },
      type: 'group',
      fields: weeks.map((week, i) => ({
        name: week,
        label: week,
        type: 'checkbox',
        admin: {
          description: weekDescriptions[i],
        },
        defaultValue: false,
        // Users can check/uncheck, but cannot add/remove weeks
      })),
      // The list of weeks is fixed in code, not editable by users
    },
  ],
}

export default Participants
