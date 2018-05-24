#pragma once

#include <math.h>

class AvgUint
{
public:
    static const unsigned int s_MaxSampleCount = 32;

private:
    unsigned int m_SampleCount;
    unsigned int* m_Samples;

    unsigned int m_AverageValue;
    unsigned int m_StandardDeviation;
    unsigned int m_StandardDeviationPercentage;
    unsigned int m_TimingVariancePercentage;

public:
    AvgUint()
    {
        m_SampleCount = 0;
        m_Samples = new unsigned int[ s_MaxSampleCount ];

        m_AverageValue = 0;
        m_StandardDeviation = 0;
    }

    ~AvgUint()
    {
        delete [] m_Samples;
    }

    void AddSample( unsigned int sample )
    {
        m_Samples[ m_SampleCount++ ] = sample;

        if ( m_SampleCount >= s_MaxSampleCount )
        {
            unsigned int sum = 0;
            for ( unsigned int i = 0; i < m_SampleCount; ++i )
                sum += m_Samples[ i ];
            m_AverageValue = static_cast< unsigned int >( roundf( static_cast< float >( sum ) / static_cast< float >( m_SampleCount ) ) );

            sum = 0;
            for ( unsigned int i = 0; i < m_SampleCount; ++i )
                sum += ( m_Samples[ i ] - m_AverageValue ) * ( m_Samples[ i ] - m_AverageValue );
            m_StandardDeviation = static_cast< unsigned int >( roundf( sqrtf( static_cast< float >( sum ) / static_cast< float >( m_SampleCount ) ) ) );
            m_StandardDeviationPercentage = static_cast< unsigned int >( roundf( 100.0f * static_cast< float >( m_StandardDeviation ) / static_cast< float >( m_AverageValue ) ) );

            unsigned int max_variance = 0;
            for ( unsigned int i = 0; i < m_SampleCount; ++i )
                max_variance = max( max_variance, static_cast< unsigned int >( abs( static_cast< int >( m_Samples[ i ] ) - static_cast< int >( m_AverageValue ) ) ) );
            m_TimingVariancePercentage = static_cast< unsigned int >( roundf( 100.0f * static_cast< float >( max_variance ) / static_cast< float >( m_AverageValue ) ) );

            m_SampleCount = 0;
        }
    }

    void Reset()
    {
        m_SampleCount = 0;
        m_AverageValue = 0;
        m_StandardDeviation = 0;
    }

    unsigned int GetAverageValue()
    {
        return m_AverageValue;
    }

    unsigned int* GetAveragePtr()
    {
        return &m_AverageValue;
    }

    unsigned int GetStandardDeviationValue()
    {
        return m_StandardDeviation;
    }

    unsigned int* GetStandardDeviationPtr()
    {
        return &m_StandardDeviation;
    }

    unsigned int GetStandardDeviationPercentageValue()
    {
        return m_StandardDeviationPercentage;
    }

    unsigned int* GetStandardDeviationPercentagePtr()
    {
        return &m_StandardDeviationPercentage;
    }

    unsigned int GetTimingVariancePercentage()
    {
        return m_TimingVariancePercentage;
    }

    unsigned int* GetTimingVariancePercentagePtr()
    {
        return &m_TimingVariancePercentage;
    }
};