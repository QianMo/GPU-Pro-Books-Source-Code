
/*

Copyright 2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901


*/


//https://code.google.com/p/thrust/source/browse/examples/repeated_range.cu
template <typename Iterator>
class repeated_range
{
    public:
    typedef typename thrust::iterator_difference<Iterator>::type difference_type;
    struct repeat_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type repeats;
        repeat_functor(difference_type repeats)
            : repeats(repeats) {}
        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return i / repeats;
        }
    };
    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<repeat_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;
    // type of the repeated_range iterator
    typedef PermutationIterator iterator;
    // construct repeated_range for the range [first,last)
    repeated_range(Iterator first, Iterator last, difference_type repeats)
        : first(first), last(last), repeats(repeats) {}
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), repeat_functor(repeats)));
    }
    iterator end(void) const
    {
        return begin() + repeats * (last - first);
    }
    protected:
    Iterator first;
    Iterator last;
    difference_type repeats;
};

//https://github.com/thrust/thrust/blob/master/examples/tiled_range.cu
template <typename Iterator>
class tiled_range
{
    public:
    typedef typename thrust::iterator_difference<Iterator>::type difference_type;
    struct tile_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type tile_size;
        tile_functor(difference_type tile_size)
            : tile_size(tile_size) {}
        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return i % tile_size;
        }
    };
    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<tile_functor, CountingIterator>   TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;
    // type of the tiled_range iterator
    typedef PermutationIterator iterator;
    // construct repeated_range for the range [first,last)
    tiled_range(Iterator first, Iterator last, difference_type tiles)
        : first(first), last(last), tiles(tiles) {}
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
    }
    iterator end(void) const
    {
        return begin() + tiles * (last - first);
    }
    protected:
    Iterator first;
    Iterator last;
    difference_type tiles;
};

/*
//https://code.google.com/p/thrust/source/browse/examples/histogram.cu
// sparse histogram using reduce_by_key
template <typename Vector1, typename Vector2, typename Vector3>
void sparse_histogram( const Vector1& input, Vector2& histogram_values, Vector3& histogram_counts )
{
	typedef typename Vector1::value_type ValueType; // input value type
	typedef typename Vector3::value_type IndexType; // histogram index type

	// copy input data (could be skipped if input is allowed to be modified)
	thrust::device_vector<ValueType> data(input);

	// sort data to bring equal elements together
	thrust::sort(data.begin(), data.end());

	// number of histogram bins is equal to number of unique values (assumes data.size() > 0)
	IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
		data.begin() + 1,
		IndexType(1),
		thrust::plus<IndexType>(),
		thrust::not_equal_to<ValueType>());

	// resize histogram storage
	histogram_values.resize(num_bins);
	histogram_counts.resize(num_bins);

	// compact find the end of each bin of values
	thrust::reduce_by_key(data.begin(), data.end(),
		thrust::constant_iterator<IndexType>(1),
		histogram_values.begin(),
		histogram_counts.begin());  
}
*/


template <typename V1>
struct GenericCopyFunctor : thrust::unary_function<V1, V1>
{
	GenericCopyFunctor(){}

	__device__ V1 operator()(V1 i)
	{
		return i;
	}
};

//https://code.google.com/p/thrust/source/browse/examples/histogram.cu
// dense histogram using binary search
template <typename Vector1, typename Vector2>
void dense_histogram( Vector1& input, Vector2& histogram )
{
	typedef typename Vector1::value_type ValueType; // input value type
	typedef typename Vector2::value_type IndexType; // histogram index type

	// copy input data (could be skipped if input is allowed to be modified)
	thrust::device_vector<ValueType> data(input);

	// sort data to bring equal elements together
	thrust::sort( data.begin(), data.end() );

	// number of histogram bins is equal to the maximum value plus one
	IndexType num_bins = data.back() + 1;

	// resize histogram storage
	histogram.resize(num_bins);

	// find the end of each bin of values
	thrust::counting_iterator<IndexType> search_begin( 0 );
	thrust::upper_bound(	data.begin(), 
							data.end(),
							search_begin, 
							search_begin + num_bins,
							histogram.begin()		);

	// compute the histogram by taking differences of the cumulative histogram
	thrust::adjacent_difference(	histogram.begin(), 
									histogram.end(),
									histogram.begin()	);
}
