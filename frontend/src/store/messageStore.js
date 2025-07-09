import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useMessageStore = defineStore('message', () => {
  const message = ref('')
  const type = ref('success') // success | error | warning | info

  function setMessage(msg, msgType = 'success') {
    message.value = msg
    type.value = msgType
  }

  return { message, type, setMessage }
})
