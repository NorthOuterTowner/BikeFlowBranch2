// src/utils/mapStyles.js
import Style from 'ol/style/Style'
import Icon from 'ol/style/Icon'
import Text from 'ol/style/Text'
import Fill from 'ol/style/Fill'

export function getStationStyle(bikeNum = 0) {
  let iconSrc = '/icons/BlueLocationRound.svg'
  if (bikeNum > 10) iconSrc = '/icons/RedLocationRound.svg'
  else if (bikeNum > 9) iconSrc = '/icons/YellowLocationRound.svg'

  return new Style({
    image: new Icon({
      src: iconSrc,
      scale: 1.5,
      anchor: [0.5, 1]
    }),
    text: new Text({
      text: bikeNum.toString(),
      fill: new Fill({ color: '#ffffff' }),
      font: '12px Arial',
      offsetY: -20
    })
  })
}
