import z from 'zod/v3'

const nameCountPrompt = {
  name: 'nameCount',
  title: 'Name Counter Prompt',
  description: 'Creates a prompt for counting names in the participants resource',
  argsSchema: {
    name:
      z.string().describe('The name to count') ||
      z.array(z.string()).describe('The names to count'),
  } as unknown as z.ZodRawShape,
  handler: ({ name }: { name: string | string[] }) => {
    const names = Array.isArray(name) ? name : [name]
    return {
      messages: [
        {
          content: {
            type: 'text',
            text: `Please count the occurrences of the following names: ${names.join(', ')}`,
          },
          role: 'user',
        },
        {
          role: 'user',
          content: {
            type: 'resource',
            resource: {
              uri: 'course://participants.md',
            },
          },
        },
      ],
    }
  },
}

export default nameCountPrompt
