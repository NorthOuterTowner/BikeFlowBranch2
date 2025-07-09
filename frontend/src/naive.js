// naive.js
import { create } from 'naive-ui'

export function createNaiveUi() {
  return create({
    themeOverrides: {
      common: {
        primaryColor: '#0556a7',
        primaryColorHover: '#0b5fb3',
        primaryColorPressed: '#044c8c',
        primaryColorSuppl: '#0556a7'
      }
    }
  })
}
